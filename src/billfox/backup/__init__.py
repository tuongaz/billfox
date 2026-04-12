"""Document backup modules for remote storage."""

from billfox._types import BackupResult
from billfox.backup._base import DocumentBackup
from billfox.backup.local import LocalBackup

__all__ = ["BackupResult", "DocumentBackup", "LocalBackup"]
