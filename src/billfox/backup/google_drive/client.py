"""Google Drive backup implementation."""

from __future__ import annotations

import asyncio
import io
from datetime import UTC, date, datetime
from pathlib import PurePath
from typing import Any

from billfox._types import BackupResult, Document
from billfox.backup.google_drive.auth import load_credentials


def _import_discovery() -> Any:
    """Lazily import googleapiclient.discovery."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "google-api-python-client is required for Google Drive backup. "
            "Install it with: pip install 'billfox[google-drive]'"
        ) from None
    return build


def _import_media_upload() -> Any:
    """Lazily import MediaIoBaseUpload."""
    try:
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError:
        raise ImportError(
            "google-api-python-client is required for Google Drive backup. "
            "Install it with: pip install 'billfox[google-drive]'"
        ) from None
    return MediaIoBaseUpload


class GoogleDriveBackup:
    """Backs up documents to Google Drive with date-based folder structure.

    Implements the DocumentBackup protocol. Uploads original document bytes
    to BillFox/YYYY/MM/DD/ folder structure in Google Drive.
    """

    def __init__(
        self,
        root_folder_name: str = "BillFox",
        credentials_path: str | None = None,
    ) -> None:
        self._root_folder_name = root_folder_name
        self._credentials_path = credentials_path
        self._service: Any = None
        self._root_folder_id: str | None = None

    def _get_service(self) -> Any:
        """Build and cache the Drive API service."""
        if self._service is None:
            build = _import_discovery()
            credentials = load_credentials(self._credentials_path)
            self._service = build("drive", "v3", credentials=credentials)
        return self._service

    def _find_folder(self, name: str, parent_folder_id: str) -> dict[str, Any] | None:
        """Find a folder by name in a parent folder."""
        service = self._get_service()
        query = (
            f"name='{name}' and "
            f"'{parent_folder_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        )
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, webViewLink)")
            .execute()
        )
        files = results.get("files", [])
        return files[0] if files else None

    def _create_folder(self, name: str, parent_folder_id: str | None = None) -> Any:
        """Create a folder in Google Drive."""
        service = self._get_service()
        file_metadata: dict[str, Any] = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_folder_id:
            file_metadata["parents"] = [parent_folder_id]
        return service.files().create(body=file_metadata, fields="id, name, webViewLink").execute()

    def _ensure_root_folder(self) -> str:
        """Find or create the root BillFox folder in Drive root."""
        if self._root_folder_id is not None:
            return self._root_folder_id

        service = self._get_service()
        query = (
            f"name='{self._root_folder_name}' and "
            f"'root' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        )
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, webViewLink)")
            .execute()
        )
        files = results.get("files", [])

        if files:
            self._root_folder_id = files[0]["id"]
        else:
            folder = self._create_folder(self._root_folder_name)
            self._root_folder_id = folder["id"]

        return self._root_folder_id

    def _ensure_folder_path(self, root_folder_id: str, path: str) -> str:
        """Ensure nested folder path exists, creating folders as needed.

        Args:
            root_folder_id: Root folder ID.
            path: Path like "2025/10/26".

        Returns:
            ID of the final (deepest) folder.
        """
        current_folder_id = root_folder_id
        for folder_name in path.split("/"):
            existing = self._find_folder(folder_name, current_folder_id)
            if existing:
                current_folder_id = existing["id"]
            else:
                new_folder = self._create_folder(folder_name, current_folder_id)
                current_folder_id = new_folder["id"]
        return current_folder_id

    def _find_file(self, file_name: str, folder_id: str) -> dict[str, Any] | None:
        """Find a file by name in a folder."""
        service = self._get_service()
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, webViewLink)")
            .execute()
        )
        files = results.get("files", [])
        return files[0] if files else None

    def _upload_or_update(
        self, content: bytes, file_name: str, mime_type: str, folder_id: str
    ) -> Any:
        """Upload a file, or update it if one with the same name already exists."""
        MediaIoBaseUpload = _import_media_upload()
        service = self._get_service()
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=True)

        existing = self._find_file(file_name, folder_id)
        if existing:
            return (
                service.files()
                .update(fileId=existing["id"], media_body=media, fields="id, name, webViewLink")
                .execute()
            )

        file_metadata: dict[str, Any] = {"name": file_name, "parents": [folder_id]}
        return (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
            .execute()
        )

    @staticmethod
    def _original_file_name(file_name: str) -> str:
        """Insert '_original' before the file extension."""
        p = PurePath(file_name)
        return f"{p.stem}_original{p.suffix}" if p.suffix else f"{file_name}_original"

    def _backup_sync(
        self,
        document: Document,
        *,
        original: Document | None = None,
        document_date: date | None = None,
    ) -> BackupResult:
        """Synchronous backup logic — runs all Drive API calls."""
        root_id = self._ensure_root_folder()

        d = document_date or datetime.now(UTC).date()
        date_path = f"{d.year}/{d.month:02d}/{d.day:02d}"
        folder_id = self._ensure_folder_path(root_id, date_path)

        file_name = PurePath(document.source_uri).name or "document"
        result = self._upload_or_update(document.content, file_name, document.mime_type, folder_id)

        original_uri: str | None = None
        if original is not None:
            original_name = self._original_file_name(file_name)
            original_result = self._upload_or_update(
                original.content, original_name, original.mime_type, folder_id,
            )
            original_uri = original_result.get("webViewLink", "")

        return BackupResult(
            uri=result.get("webViewLink", ""),
            provider="google_drive",
            original_uri=original_uri,
            metadata={"file_id": result.get("id", ""), "file_name": result.get("name", "")},
        )

    async def backup(
        self,
        document: Document,
        *,
        original: Document | None = None,
        document_date: date | None = None,
    ) -> BackupResult:
        """Back up a document to Google Drive.

        Creates a date-based folder structure (BillFox/YYYY/MM/DD/) using the
        *document_date* (e.g. receipt expense date). Falls back to today
        when *document_date* is not provided.

        When *original* is provided, it is also uploaded alongside the main
        file with an ``_original`` suffix.

        Returns:
            BackupResult with the Google Drive webViewLink as uri.
        """
        return await asyncio.to_thread(
            self._backup_sync, document, original=original, document_date=document_date,
        )
