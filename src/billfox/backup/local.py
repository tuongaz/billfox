"""Local filesystem backup implementation."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path, PurePath

from billfox._types import BackupResult, Document


class LocalBackup:
    """Backs up documents to a local folder with date-based structure.

    Implements the DocumentBackup protocol. Copies original document bytes
    to {base_path}/YYYY/MM/DD/ folder structure.
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)

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
        """Synchronous backup logic."""
        d = document_date or datetime.now(UTC).date()
        date_folder = self._base_path / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}"
        date_folder.mkdir(parents=True, exist_ok=True)

        file_name = PurePath(document.source_uri).name or "document"
        target_path = date_folder / file_name
        target_path.write_bytes(document.content)

        original_uri: str | None = None
        if original is not None:
            original_name = self._original_file_name(file_name)
            original_path = date_folder / original_name
            original_path.write_bytes(original.content)
            original_uri = str(original_path)

        return BackupResult(
            uri=str(target_path),
            provider="local",
            original_uri=original_uri,
            metadata={"file_name": file_name},
        )

    async def backup(
        self,
        document: Document,
        *,
        original: Document | None = None,
        document_date: date | None = None,
    ) -> BackupResult:
        """Back up a document to the local filesystem.

        Creates a date-based folder structure (YYYY/MM/DD/) using the
        *document_date* (e.g. receipt expense date). Falls back to today
        when *document_date* is not provided.

        When *original* is provided, it is also saved alongside the main
        file with an ``_original`` suffix.

        Returns:
            BackupResult with the local file path as uri.
        """
        return await asyncio.to_thread(
            self._backup_sync, document, original=original, document_date=document_date,
        )
