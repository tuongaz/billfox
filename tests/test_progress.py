from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from billfox._progress import ProgressEvent, Stage, Status
from billfox._types import Document, ExtractionResult, Page
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
def mock_preprocessor(preprocessed_document: Document) -> AsyncMock:
    preprocessor = AsyncMock()
    preprocessor.process = AsyncMock(return_value=preprocessed_document)
    return preprocessor


class TestProgressNoCallback:
    """Pipeline with no on_progress behaves normally."""

    @pytest.mark.asyncio
    async def test_run_without_on_progress(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        parsed_invoice: Invoice,
    ) -> None:
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.run("test.png")

        assert result == parsed_invoice

    @pytest.mark.asyncio
    async def test_extract_only_without_on_progress(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        extraction_result: ExtractionResult,
    ) -> None:
        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.extract_only("test.png")

        assert result == extraction_result


class TestProgressRunEvents:
    """run() emits the correct sequence of progress events."""

    @pytest.mark.asyncio
    async def test_run_emits_all_stage_events(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
        mock_preprocessor: AsyncMock,
    ) -> None:
        """run() emits STARTED/COMPLETED for all stages."""
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            store=mock_store,
            on_progress=on_progress,
        )

        await pipeline.run("test.png", document_id="doc-1")

        stages_seen = [(e.stage, e.status) for e in events]
        assert stages_seen == [
            (Stage.LOADING, Status.STARTED),
            (Stage.LOADING, Status.COMPLETED),
            (Stage.PREPROCESSING, Status.STARTED),
            (Stage.PREPROCESSING, Status.COMPLETED),
            (Stage.EXTRACTING, Status.STARTED),
            (Stage.EXTRACTING, Status.COMPLETED),
            (Stage.PARSING, Status.STARTED),
            (Stage.PARSING, Status.COMPLETED),
            (Stage.STORING, Status.STARTED),
            (Stage.STORING, Status.COMPLETED),
        ]


class TestProgressExtractOnlyEvents:
    """extract_only() emits only LOADING, PREPROCESSING, EXTRACTING events."""

    @pytest.mark.asyncio
    async def test_extract_only_emits_no_parsing_or_storing(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            on_progress=on_progress,
        )

        await pipeline.extract_only("test.png")

        stages_seen = [(e.stage, e.status) for e in events]
        assert stages_seen == [
            (Stage.LOADING, Status.STARTED),
            (Stage.LOADING, Status.COMPLETED),
            (Stage.PREPROCESSING, Status.STARTED),
            (Stage.PREPROCESSING, Status.COMPLETED),
            (Stage.EXTRACTING, Status.STARTED),
            (Stage.EXTRACTING, Status.COMPLETED),
        ]

        emitted_stages = {e.stage for e in events}
        assert Stage.PARSING not in emitted_stages
        assert Stage.STORING not in emitted_stages


class TestProgressFailedEvents:
    """FAILED events are emitted when a stage raises an exception."""

    @pytest.mark.asyncio
    async def test_loading_failure_emits_failed_event(
        self,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        source = AsyncMock()
        source.load = AsyncMock(side_effect=FileNotFoundError("file not found"))

        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=source,
            extractor=mock_extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        with pytest.raises(FileNotFoundError):
            await pipeline.run("missing.png")

        failed_events = [e for e in events if e.status == Status.FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].stage == Stage.LOADING
        assert failed_events[0].message == "file not found"

    @pytest.mark.asyncio
    async def test_extracting_failure_emits_failed_event(
        self,
        mock_source: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        extractor = AsyncMock()
        extractor.extract = AsyncMock(side_effect=RuntimeError("extraction failed"))

        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        with pytest.raises(RuntimeError):
            await pipeline.run("test.png")

        failed_events = [e for e in events if e.status == Status.FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].stage == Stage.EXTRACTING
        assert failed_events[0].message == "extraction failed"

    @pytest.mark.asyncio
    async def test_parsing_failure_emits_failed_event(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> None:
        parser = AsyncMock()
        parser.parse = AsyncMock(side_effect=ValueError("parse error"))

        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=parser,
            on_progress=on_progress,
        )

        with pytest.raises(ValueError):
            await pipeline.run("test.png")

        failed_events = [e for e in events if e.status == Status.FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].stage == Stage.PARSING
        assert failed_events[0].message == "parse error"

    @pytest.mark.asyncio
    async def test_storing_failure_emits_failed_event(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        store = AsyncMock()
        store.save = AsyncMock(side_effect=IOError("store failed"))

        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=store,
            on_progress=on_progress,
        )

        with pytest.raises(IOError):
            await pipeline.run("test.png", document_id="doc-1")

        failed_events = [e for e in events if e.status == Status.FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].stage == Stage.STORING
        assert failed_events[0].message == "store failed"

    @pytest.mark.asyncio
    async def test_extract_only_failure_emits_failed_event(
        self,
        mock_source: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        extractor = AsyncMock()
        extractor.extract = AsyncMock(side_effect=RuntimeError("extract error"))

        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        with pytest.raises(RuntimeError):
            await pipeline.extract_only("test.png")

        failed_events = [e for e in events if e.status == Status.FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].stage == Stage.EXTRACTING
        assert failed_events[0].message == "extract error"


class TestProgressConditionalEvents:
    """Events are skipped when stages are not applicable."""

    @pytest.mark.asyncio
    async def test_preprocessing_skipped_when_no_preprocessors(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        await pipeline.run("test.png")

        emitted_stages = {e.stage for e in events}
        assert Stage.PREPROCESSING not in emitted_stages

    @pytest.mark.asyncio
    async def test_storing_skipped_when_no_store(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=None,
            on_progress=on_progress,
        )

        await pipeline.run("test.png", document_id="doc-1")

        emitted_stages = {e.stage for e in events}
        assert Stage.STORING not in emitted_stages

    @pytest.mark.asyncio
    async def test_storing_skipped_when_no_document_id(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_store: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=mock_store,
            on_progress=on_progress,
        )

        await pipeline.run("test.png")

        emitted_stages = {e.stage for e in events}
        assert Stage.STORING not in emitted_stages


class TestProgressEventMetadata:
    """Progress events include correct message and metadata fields."""

    @pytest.mark.asyncio
    async def test_preprocessing_started_includes_class_name(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            on_progress=on_progress,
        )

        await pipeline.run("test.png")

        preprocessing_started = [
            e for e in events if e.stage == Stage.PREPROCESSING and e.status == Status.STARTED
        ]
        assert len(preprocessing_started) == 1
        assert preprocessing_started[0].message == "AsyncMock"

    @pytest.mark.asyncio
    async def test_extracting_completed_includes_page_count(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        await pipeline.run("test.png")

        extracting_completed = [
            e for e in events if e.stage == Stage.EXTRACTING and e.status == Status.COMPLETED
        ]
        assert len(extracting_completed) == 1
        assert extracting_completed[0].metadata == {"pages": 2}

    @pytest.mark.asyncio
    async def test_extract_only_extracting_completed_includes_page_count(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        events: list[ProgressEvent] = []

        async def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline: Pipeline[Invoice] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            on_progress=on_progress,
        )

        await pipeline.extract_only("test.png")

        extracting_completed = [
            e for e in events if e.stage == Stage.EXTRACTING and e.status == Status.COMPLETED
        ]
        assert len(extracting_completed) == 1
        assert extracting_completed[0].metadata == {"pages": 2}
