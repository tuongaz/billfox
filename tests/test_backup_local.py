"""Tests for LocalBackup provider."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from billfox._types import BackupResult, Document
from billfox.backup._base import DocumentBackup
from billfox.backup.local import LocalBackup


@pytest.fixture
def sample_document() -> Document:
    return Document(
        content=b"fake invoice bytes",
        mime_type="application/pdf",
        source_uri="/path/to/invoice.pdf",
    )


# ---------------------------------------------------------------------------
# TestLocalBackupProtocol
# ---------------------------------------------------------------------------


class TestLocalBackupProtocol:
    """LocalBackup satisfies the DocumentBackup protocol."""

    def test_isinstance_check(self, tmp_path: Path) -> None:
        backup = LocalBackup(base_path=str(tmp_path))
        assert isinstance(backup, DocumentBackup)


# ---------------------------------------------------------------------------
# TestLocalBackupFolderCreation
# ---------------------------------------------------------------------------


class TestLocalBackupFolderCreation:
    """LocalBackup creates the correct date-based folder structure."""

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_creates_date_folder_structure(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
        sample_document: Document,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 3, 7, 12, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        backup = LocalBackup(base_path=str(tmp_path))
        result = await backup.backup(sample_document)

        expected_dir = tmp_path / "2025" / "03" / "07"
        assert expected_dir.is_dir()
        assert result.uri == str(expected_dir / "invoice.pdf")

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_zero_padded_month_and_day(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
        sample_document: Document,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 1, 5, 8, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        backup = LocalBackup(base_path=str(tmp_path))
        result = await backup.backup(sample_document)

        expected_dir = tmp_path / "2025" / "01" / "05"
        assert expected_dir.is_dir()


# ---------------------------------------------------------------------------
# TestLocalBackupFileWriting
# ---------------------------------------------------------------------------


class TestLocalBackupFileWriting:
    """LocalBackup writes document content correctly."""

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_writes_document_bytes(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
        sample_document: Document,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        backup = LocalBackup(base_path=str(tmp_path))
        result = await backup.backup(sample_document)

        target = Path(result.uri)
        assert target.exists()
        assert target.read_bytes() == b"fake invoice bytes"

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_extracts_filename_from_source_uri(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        doc = Document(
            content=b"data",
            mime_type="image/png",
            source_uri="/some/nested/path/receipt.png",
        )
        backup = LocalBackup(base_path=str(tmp_path))
        result = await backup.backup(doc)

        assert Path(result.uri).name == "receipt.png"


# ---------------------------------------------------------------------------
# TestLocalBackupOverwrite
# ---------------------------------------------------------------------------


class TestLocalBackupOverwrite:
    """LocalBackup overwrites existing files with the same name."""

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_overwrites_existing_file(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        backup = LocalBackup(base_path=str(tmp_path))

        doc_v1 = Document(content=b"version 1", mime_type="application/pdf", source_uri="/invoice.pdf")
        doc_v2 = Document(content=b"version 2", mime_type="application/pdf", source_uri="/invoice.pdf")

        await backup.backup(doc_v1)
        result = await backup.backup(doc_v2)

        target = Path(result.uri)
        assert target.read_bytes() == b"version 2"


# ---------------------------------------------------------------------------
# TestLocalBackupResult
# ---------------------------------------------------------------------------


class TestLocalBackupResult:
    """LocalBackup returns correct BackupResult."""

    @patch("billfox.backup.local.datetime")
    @pytest.mark.asyncio
    async def test_result_fields(
        self,
        mock_datetime: pytest.fixture,
        tmp_path: Path,
        sample_document: Document,
    ) -> None:
        mock_datetime.now.return_value = datetime(2025, 3, 7, 12, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        backup = LocalBackup(base_path=str(tmp_path))
        result = await backup.backup(sample_document)

        assert isinstance(result, BackupResult)
        assert result.provider == "local"
        assert result.metadata == {"file_name": "invoice.pdf"}
        assert result.uri.endswith("invoice.pdf")
