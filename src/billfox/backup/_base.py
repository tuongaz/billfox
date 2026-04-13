from __future__ import annotations

from typing import Protocol, runtime_checkable

from billfox._types import BackupResult, Document


@runtime_checkable
class DocumentBackup(Protocol):
    """Protocol for backing up documents to a remote location."""

    async def backup(self, document: Document, *, original: Document | None = None) -> BackupResult: ...
