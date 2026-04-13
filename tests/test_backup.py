from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from billfox._types import BackupResult, Document, ExtractionResult, Page
from billfox.backup._base import DocumentBackup
from billfox.backup.google_drive.client import GoogleDriveBackup
from billfox.pipeline import Pipeline


class Invoice(BaseModel):
    vendor_name: str
    total: float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    return Document(
        content=b"fake image bytes",
        mime_type="image/png",
        source_uri="/path/to/invoice.png",
    )


@pytest.fixture
def extraction_result() -> ExtractionResult:
    return ExtractionResult(
        markdown="# Invoice\nVendor: Acme\nTotal: $100.00",
        pages=[
            Page(index=0, markdown="page 0"),
            Page(index=1, markdown="page 1"),
        ],
    )


@pytest.fixture
def parsed_invoice() -> Invoice:
    return Invoice(vendor_name="Acme", total=100.0)


@pytest.fixture
def mock_source(sample_document: Document) -> AsyncMock:
    source = AsyncMock()
    source.load = AsyncMock(return_value=sample_document)
    return source


@pytest.fixture
def mock_extractor(extraction_result: ExtractionResult) -> AsyncMock:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=extraction_result)
    return extractor


@pytest.fixture
def mock_parser(parsed_invoice: Invoice) -> AsyncMock:
    parser = AsyncMock()
    parser.parse = AsyncMock(return_value=parsed_invoice)
    return parser


@pytest.fixture
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_backup() -> AsyncMock:
    backup = AsyncMock()
    backup.backup = AsyncMock(
        return_value=BackupResult(
            uri="https://drive.google.com/file/abc",
            provider="google_drive",
            metadata={"file_id": "abc", "file_name": "invoice.png"},
        )
    )
    return backup


# ---------------------------------------------------------------------------
# TestBackupResultDataclass
# ---------------------------------------------------------------------------


class TestBackupResultDataclass:
    """BackupResult is a frozen dataclass with correct fields."""

    def test_fields_are_correct(self) -> None:
        result = BackupResult(
            uri="https://drive.google.com/file/abc",
            provider="google_drive",
            metadata={"file_id": "abc"},
        )
        assert result.uri == "https://drive.google.com/file/abc"
        assert result.provider == "google_drive"
        assert result.metadata == {"file_id": "abc"}

    def test_frozen(self) -> None:
        result = BackupResult(uri="x", provider="y")
        with pytest.raises(FrozenInstanceError):
            result.uri = "z"  # type: ignore[misc]

    def test_default_metadata(self) -> None:
        result = BackupResult(uri="x", provider="y")
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# TestDocumentBackupProtocol
# ---------------------------------------------------------------------------


class TestDocumentBackupProtocol:
    """GoogleDriveBackup satisfies the DocumentBackup protocol."""

    def test_isinstance_check(self) -> None:
        backup = GoogleDriveBackup()
        assert isinstance(backup, DocumentBackup)


# ---------------------------------------------------------------------------
# TestPipelineBackupIntegration
# ---------------------------------------------------------------------------


class TestPipelineBackupIntegration:
    """Pipeline backup integration tests."""

    @pytest.mark.asyncio
    async def test_run_calls_backup_after_completion(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
        mock_backup: AsyncMock,
        sample_document: Document,
        parsed_invoice: Invoice,
    ) -> None:
        """Pipeline.run() calls backup.backup(document) after all stages."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=mock_store,
            backup=mock_backup,
        )

        result = await pipeline.run("test.png", document_id="doc-1")

        assert result == parsed_invoice
        mock_store.save.assert_called_once()
        mock_backup.backup.assert_called_once_with(sample_document, original=None, document_date=None)

    @pytest.mark.asyncio
    async def test_extract_only_calls_backup(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_backup: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Pipeline.extract_only() also triggers backup."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            backup=mock_backup,
        )

        await pipeline.extract_only("test.png")

        mock_backup.backup.assert_called_once_with(sample_document, original=None)

    @pytest.mark.asyncio
    async def test_pipeline_continues_when_backup_fails(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        parsed_invoice: Invoice,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Backup failure does not prevent the pipeline from completing."""
        failing_backup = AsyncMock()
        failing_backup.backup = AsyncMock(side_effect=RuntimeError("Drive API error"))

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            backup=failing_backup,
        )

        with caplog.at_level(logging.WARNING, logger="billfox.pipeline"):
            result = await pipeline.run("test.png")

        assert result == parsed_invoice
        assert "Backup failed" in caplog.text

    @pytest.mark.asyncio
    async def test_extract_only_continues_when_backup_fails(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        extraction_result: ExtractionResult,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """extract_only() also survives backup failure."""
        failing_backup = AsyncMock()
        failing_backup.backup = AsyncMock(side_effect=RuntimeError("Drive API error"))

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            backup=failing_backup,
        )

        with caplog.at_level(logging.WARNING, logger="billfox.pipeline"):
            result = await pipeline.extract_only("test.png")

        assert result == extraction_result
        assert "Backup failed" in caplog.text

    @pytest.mark.asyncio
    async def test_pipeline_without_backup_unchanged(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        parsed_invoice: Invoice,
    ) -> None:
        """Pipeline with no backup behaves identically to before."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.run("test.png")

        assert result == parsed_invoice


# ---------------------------------------------------------------------------
# TestGoogleDriveBackupFolderPath
# ---------------------------------------------------------------------------


class TestGoogleDriveBackupFolderPath:
    """GoogleDriveBackup constructs the correct date-based folder path."""

    @patch("billfox.backup.google_drive.client.load_credentials")
    @patch("billfox.backup.google_drive.client._import_discovery")
    @patch("billfox.backup.google_drive.client._import_media_upload")
    @patch("billfox.backup.google_drive.client.datetime")
    def test_folder_path_for_date(
        self,
        mock_datetime: MagicMock,
        mock_media_upload: MagicMock,
        mock_discovery: MagicMock,
        mock_load_creds: MagicMock,
    ) -> None:
        """Correct YYYY/MM/DD folder path is constructed from UTC date."""
        from datetime import UTC, datetime

        mock_datetime.now.return_value = datetime(2025, 3, 7, 12, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        mock_creds = MagicMock()
        mock_load_creds.return_value = mock_creds

        mock_service = MagicMock()
        mock_build = MagicMock(return_value=mock_service)
        mock_discovery.return_value = mock_build

        # Mock files().list() for _ensure_root_folder — root folder exists
        root_list_result = {"files": [{"id": "root-id", "name": "BillFox", "webViewLink": "link"}]}
        # Mock files().list() for _find_folder calls (year, month, day)
        year_result = {"files": [{"id": "year-id", "name": "2025", "webViewLink": "link"}]}
        month_result = {"files": [{"id": "month-id", "name": "03", "webViewLink": "link"}]}
        day_result: dict[str, list[dict[str, str]]] = {"files": []}  # Day folder doesn't exist

        list_mock = mock_service.files.return_value.list
        list_mock.return_value.execute.side_effect = [
            root_list_result,
            year_result,
            month_result,
            day_result,
        ]

        # Mock folder creation for the day folder
        create_mock = mock_service.files.return_value.create
        create_mock.return_value.execute.side_effect = [
            {"id": "day-id", "name": "07", "webViewLink": "link"},
            {"id": "file-id", "name": "invoice.png", "webViewLink": "https://drive.google.com/file/xyz"},
        ]

        # Mock _find_file returning None (no existing file)
        # The 5th list call is for _find_file
        list_mock.return_value.execute.side_effect = [
            root_list_result,
            year_result,
            month_result,
            day_result,
            {"files": []},  # _find_file — no existing file
        ]

        # Mock MediaIoBaseUpload
        mock_media_cls = MagicMock()
        mock_media_upload.return_value = mock_media_cls

        backup = GoogleDriveBackup()
        doc = Document(content=b"bytes", mime_type="image/png", source_uri="/path/to/invoice.png")
        result = backup._backup_sync(doc)

        assert result.provider == "google_drive"
        assert result.uri == "https://drive.google.com/file/xyz"

        # Verify folder creation was called with day "07" (the missing folder)
        create_calls = create_mock.call_args_list
        day_creation = create_calls[0]
        body_arg = day_creation[1]["body"]
        assert body_arg["name"] == "07"


# ---------------------------------------------------------------------------
# TestGoogleDriveBackupDeduplication
# ---------------------------------------------------------------------------


class TestGoogleDriveBackupDeduplication:
    """File deduplication — update instead of create when file exists."""

    @patch("billfox.backup.google_drive.client.load_credentials")
    @patch("billfox.backup.google_drive.client._import_discovery")
    @patch("billfox.backup.google_drive.client._import_media_upload")
    @patch("billfox.backup.google_drive.client.datetime")
    def test_updates_existing_file(
        self,
        mock_datetime: MagicMock,
        mock_media_upload: MagicMock,
        mock_discovery: MagicMock,
        mock_load_creds: MagicMock,
    ) -> None:
        """When a file with the same name exists, update instead of create."""
        from datetime import UTC, datetime

        mock_datetime.now.return_value = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC

        mock_creds = MagicMock()
        mock_load_creds.return_value = mock_creds

        mock_service = MagicMock()
        mock_build = MagicMock(return_value=mock_service)
        mock_discovery.return_value = mock_build

        root_list_result = {"files": [{"id": "root-id", "name": "BillFox", "webViewLink": "link"}]}
        year_result = {"files": [{"id": "year-id", "name": "2025", "webViewLink": "link"}]}
        month_result = {"files": [{"id": "month-id", "name": "06", "webViewLink": "link"}]}
        day_result = {"files": [{"id": "day-id", "name": "15", "webViewLink": "link"}]}
        # Existing file found — deduplication triggers update
        existing_file = {"files": [{"id": "existing-id", "name": "receipt.pdf", "webViewLink": "old-link"}]}

        list_mock = mock_service.files.return_value.list
        list_mock.return_value.execute.side_effect = [
            root_list_result,
            year_result,
            month_result,
            day_result,
            existing_file,  # _find_file returns existing file
        ]

        # Mock update (not create)
        update_mock = mock_service.files.return_value.update
        update_mock.return_value.execute.return_value = {
            "id": "existing-id",
            "name": "receipt.pdf",
            "webViewLink": "https://drive.google.com/file/updated",
        }

        mock_media_cls = MagicMock()
        mock_media_upload.return_value = mock_media_cls

        backup = GoogleDriveBackup()
        doc = Document(content=b"pdf bytes", mime_type="application/pdf", source_uri="/docs/receipt.pdf")
        result = backup._backup_sync(doc)

        assert result.uri == "https://drive.google.com/file/updated"
        # Verify update was called (not create for the file)
        update_mock.assert_called_once()
        update_call = update_mock.call_args
        assert update_call[1]["fileId"] == "existing-id"


# ---------------------------------------------------------------------------
# TestCredentialsLoading
# ---------------------------------------------------------------------------


class TestCredentialsLoading:
    """Credentials loading and missing-credentials error message."""

    def test_missing_credentials_raises_file_not_found(self, tmp_path: Path) -> None:
        """Clear error when credentials file is missing."""
        from billfox.backup.google_drive.auth import load_credentials

        nonexistent = str(tmp_path / "nonexistent.json")

        with pytest.raises(FileNotFoundError, match="Run 'billfox auth google-drive' to authorize"):
            load_credentials(nonexistent)

    @patch("billfox.backup.google_drive.auth._import_google_auth")
    def test_load_credentials_reads_file(
        self,
        mock_import_auth: MagicMock,
        tmp_path: Path,
    ) -> None:
        """load_credentials reads and returns a Credentials object."""
        import json

        from billfox.backup.google_drive.auth import load_credentials

        mock_creds_cls = MagicMock()
        mock_creds_instance = MagicMock()
        mock_creds_instance.expired = False
        mock_creds_cls.return_value = mock_creds_instance
        mock_import_auth.return_value = mock_creds_cls

        cred_data = {
            "access_token": "token123",
            "refresh_token": "refresh456",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
        }
        cred_file = tmp_path / "google_drive.json"
        cred_file.write_text(json.dumps(cred_data))

        result = load_credentials(str(cred_file))

        assert result == mock_creds_instance
        mock_creds_cls.assert_called_once_with(
            token="token123",
            refresh_token="refresh456",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="client-id",
            client_secret="client-secret",
        )
