"""Local filesystem backup implementation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path, PurePath

from billfox._types import BackupResult, Document


class LocalBackup:
    """Backs up documents to a local folder with date-based structure.

    Implements the DocumentBackup protocol. Copies original document bytes
    to {base_path}/YYYY/MM/DD/ folder structure.
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)

    def _backup_sync(self, document: Document) -> BackupResult:
        """Synchronous backup logic."""
        now = datetime.now(UTC)
        date_folder = self._base_path / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}"
        date_folder.mkdir(parents=True, exist_ok=True)

        file_name = PurePath(document.source_uri).name or "document"
        target_path = date_folder / file_name
        target_path.write_bytes(document.content)

        return BackupResult(
            uri=str(target_path),
            provider="local",
            metadata={"file_name": file_name},
        )

    async def backup(self, document: Document) -> BackupResult:
        """Back up a document to the local filesystem.

        Creates a date-based folder structure (YYYY/MM/DD/) and writes
        the original document bytes. If a file with the same name exists
        in the target folder, it is overwritten.

        Returns:
            BackupResult with the local file path as uri.
        """
        return await asyncio.to_thread(self._backup_sync, document)
