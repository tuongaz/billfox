from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from billfox._types import BackupResult, Document, ExtractionResult, Page
from billfox.pipeline import Pipeline


class Invoice(BaseModel):
    vendor_name: str
    total: float


@pytest.fixture
def sample_document() -> Document:
    return Document(
        content=b"fake image bytes",
        mime_type="image/png",
        source_uri="test.png",
    )


@pytest.fixture
def preprocessed_document() -> Document:
    return Document(
        content=b"preprocessed bytes",
        mime_type="image/png",
        source_uri="test.png",
    )


@pytest.fixture
def extraction_result() -> ExtractionResult:
    return ExtractionResult(
        markdown="# Invoice\nVendor: Acme\nTotal: $100.00",
        pages=[Page(index=0, markdown="# Invoice\nVendor: Acme\nTotal: $100.00")],
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
    store.get = AsyncMock(return_value=None)
    store.search = AsyncMock(return_value=[])
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_preprocessor(preprocessed_document: Document) -> AsyncMock:
    preprocessor = AsyncMock()
    preprocessor.process = AsyncMock(return_value=preprocessed_document)
    return preprocessor


class TestPipelineRun:
    """Tests for Pipeline.run() method."""

    @pytest.mark.asyncio
    async def test_full_pipeline_call_order(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
        mock_preprocessor: AsyncMock,
        preprocessed_document: Document,
        extraction_result: ExtractionResult,
        parsed_invoice: Invoice,
    ) -> None:
        """All stages are called in the correct order."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            store=mock_store,
        )

        result = await pipeline.run("test.png", document_id="doc-1")

        assert result == parsed_invoice
        mock_source.load.assert_called_once_with("test.png")
        mock_preprocessor.process.assert_called_once()
        mock_extractor.extract.assert_called_once_with(preprocessed_document, on_step=None)
        mock_parser.parse.assert_called_once_with(extraction_result.markdown)
        mock_store.save.assert_called_once_with("doc-1", parsed_invoice)

    @pytest.mark.asyncio
    async def test_pipeline_without_preprocessors(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        sample_document: Document,
        parsed_invoice: Invoice,
    ) -> None:
        """Pipeline works with empty preprocessors list."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.run("test.png")

        assert result == parsed_invoice
        mock_extractor.extract.assert_called_once_with(sample_document, on_step=None)

    @pytest.mark.asyncio
    async def test_pipeline_without_store(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        parsed_invoice: Invoice,
    ) -> None:
        """Pipeline works when store is None."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=None,
        )

        result = await pipeline.run("test.png", document_id="doc-1")

        assert result == parsed_invoice

    @pytest.mark.asyncio
    async def test_store_not_called_without_document_id(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
        parsed_invoice: Invoice,
    ) -> None:
        """Store is not called when document_id is None."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=mock_store,
        )

        result = await pipeline.run("test.png")

        assert result == parsed_invoice
        mock_store.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_called_only_with_both_store_and_document_id(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
        parsed_invoice: Invoice,
    ) -> None:
        """Store.save is called only when both store and document_id are provided."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=mock_store,
        )

        result = await pipeline.run("test.png", document_id="doc-1")

        assert result == parsed_invoice
        mock_store.save.assert_called_once_with("doc-1", parsed_invoice)

    @pytest.mark.asyncio
    async def test_multiple_preprocessors_applied_in_order(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        """Multiple preprocessors are applied sequentially."""
        doc1 = Document(content=b"step1", mime_type="image/png", source_uri="test.png")
        doc2 = Document(content=b"step2", mime_type="image/png", source_uri="test.png")

        preprocessor1 = AsyncMock()
        preprocessor1.process = AsyncMock(return_value=doc1)
        preprocessor2 = AsyncMock()
        preprocessor2.process = AsyncMock(return_value=doc2)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[preprocessor1, preprocessor2],
        )

        await pipeline.run("test.png")

        preprocessor1.process.assert_called_once()
        preprocessor2.process.assert_called_once_with(doc1)
        mock_extractor.extract.assert_called_once_with(doc2, on_step=None)


class TestPipelineExtractOnly:
    """Tests for Pipeline.extract_only() method."""

    @pytest.mark.asyncio
    async def test_extract_only_returns_extraction_result(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        extraction_result: ExtractionResult,
    ) -> None:
        """extract_only returns ExtractionResult without parsing."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.extract_only("test.png")

        assert result == extraction_result

    @pytest.mark.asyncio
    async def test_extract_only_skips_parse_and_store(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
    ) -> None:
        """extract_only does not call parser or store."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=mock_store,
        )

        await pipeline.extract_only("test.png")

        mock_parser.parse.assert_not_called()
        mock_store.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_only_applies_preprocessors(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
        preprocessed_document: Document,
    ) -> None:
        """extract_only still applies preprocessors."""
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
        )

        await pipeline.extract_only("test.png")

        mock_preprocessor.process.assert_called_once()
        mock_extractor.extract.assert_called_once_with(preprocessed_document, on_step=None)


class TestPipelineBackupWithOriginal:
    """Pipeline passes the original document to backup when preprocessors are used."""

    @pytest.mark.asyncio
    async def test_backup_receives_original_when_preprocessed(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
        sample_document: Document,
        preprocessed_document: Document,
    ) -> None:
        """Backup is called with original=original_document when preprocessors run."""
        mock_backup = AsyncMock()
        mock_backup.backup = AsyncMock(
            return_value=BackupResult(uri="/backup/test.png", provider="local")
        )

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            backup=mock_backup,
        )

        await pipeline.run("test.png")

        mock_backup.backup.assert_called_once_with(preprocessed_document, original=sample_document)

    @pytest.mark.asyncio
    async def test_backup_receives_no_original_without_preprocessors(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        sample_document: Document,
    ) -> None:
        """Backup is called with original=None when no preprocessors."""
        mock_backup = AsyncMock()
        mock_backup.backup = AsyncMock(
            return_value=BackupResult(uri="/backup/test.png", provider="local")
        )

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            backup=mock_backup,
        )

        await pipeline.run("test.png")

        mock_backup.backup.assert_called_once_with(sample_document, original=None)

    @pytest.mark.asyncio
    async def test_extract_only_backup_receives_original(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
        sample_document: Document,
        preprocessed_document: Document,
    ) -> None:
        """extract_only also passes original to backup."""
        mock_backup = AsyncMock()
        mock_backup.backup = AsyncMock(
            return_value=BackupResult(uri="/backup/test.png", provider="local")
        )

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            backup=mock_backup,
        )

        await pipeline.extract_only("test.png")

        mock_backup.backup.assert_called_once_with(preprocessed_document, original=sample_document)
