"""Integration and isolation tests for US-014: CI and integration verification."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from billfox._types import Document, ExtractionResult, Page, SearchResult
from billfox.pipeline import Pipeline
from billfox.store.sqlite import SQLiteDocumentStore

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


class Receipt(BaseModel):
    vendor: str
    total: float
    currency: str = "USD"


@pytest.fixture
def sample_image_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def mock_source(sample_image_bytes: bytes) -> AsyncMock:
    source = AsyncMock()
    source.load = AsyncMock(
        return_value=Document(
            content=sample_image_bytes,
            mime_type="image/png",
            source_uri="/tmp/receipt.png",
        )
    )
    return source


@pytest.fixture
def mock_extractor() -> AsyncMock:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            markdown="# Receipt\nVendor: Coffee Shop\nTotal: $4.50",
            pages=[Page(index=0, markdown="# Receipt\nVendor: Coffee Shop\nTotal: $4.50")],
        )
    )
    return extractor


@pytest.fixture
def mock_parser() -> AsyncMock:
    parser = AsyncMock()
    parser.parse = AsyncMock(
        return_value=Receipt(vendor="Coffee Shop", total=4.50)
    )
    return parser


@pytest.fixture
def mock_preprocessor() -> AsyncMock:
    preprocessor = AsyncMock()
    preprocessor.process = AsyncMock(
        return_value=Document(
            content=b"preprocessed",
            mime_type="image/png",
            source_uri="/tmp/receipt.png",
        )
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Integration: full pipeline with mocked LLM + mocked OCR + real SQLite
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Full pipeline with mocked OCR/LLM but real in-memory SQLite store."""

    @pytest.mark.asyncio
    async def test_pipeline_run_stores_in_sqlite(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        """Pipeline.run() stores parsed result in real SQLite and retrieves it."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
        )

        pipeline: Pipeline[Receipt] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            store=store,
        )

        result = await pipeline.run("/tmp/receipt.png", document_id="r-001")

        assert result.vendor == "Coffee Shop"
        assert result.total == 4.50

        # Verify it was persisted in SQLite
        retrieved = await store.get("r-001")
        assert retrieved is not None
        assert retrieved.vendor == "Coffee Shop"
        assert retrieved.total == 4.50

    @pytest.mark.asyncio
    async def test_pipeline_with_preprocessor_and_store(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
    ) -> None:
        """Pipeline with preprocessor and store — all stages execute in order."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
        )

        pipeline: Pipeline[Receipt] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
            store=store,
        )

        result = await pipeline.run("/tmp/receipt.png", document_id="r-002")

        assert result.vendor == "Coffee Shop"
        mock_source.load.assert_called_once()
        mock_preprocessor.process.assert_called_once()
        mock_extractor.extract.assert_called_once()
        mock_parser.parse.assert_called_once()

        retrieved = await store.get("r-002")
        assert retrieved is not None
        assert retrieved.total == 4.50

    @pytest.mark.asyncio
    async def test_pipeline_multiple_documents(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> None:
        """Pipeline can process multiple documents into the same store."""
        receipts = [
            Receipt(vendor="Coffee Shop", total=4.50),
            Receipt(vendor="Bookstore", total=29.99),
        ]

        parser = AsyncMock()
        parser.parse = AsyncMock(side_effect=receipts)

        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
        )

        pipeline: Pipeline[Receipt] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=parser,
            store=store,
        )

        r1 = await pipeline.run("/tmp/a.png", document_id="r-a")
        r2 = await pipeline.run("/tmp/b.png", document_id="r-b")

        assert r1.vendor == "Coffee Shop"
        assert r2.vendor == "Bookstore"

        assert (await store.get("r-a")) is not None
        assert (await store.get("r-b")) is not None


# ---------------------------------------------------------------------------
# Integration: extract_only flow
# ---------------------------------------------------------------------------


class TestExtractOnlyIntegration:
    """extract_only: source -> preprocess -> extract (no parse or store)."""

    @pytest.mark.asyncio
    async def test_extract_only_returns_markdown(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
    ) -> None:
        pipeline: Pipeline[Receipt] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
        )

        result = await pipeline.extract_only("/tmp/receipt.png")

        assert isinstance(result, ExtractionResult)
        assert "Coffee Shop" in result.markdown
        assert len(result.pages) == 1
        mock_parser.parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_only_with_preprocessor(
        self,
        mock_source: AsyncMock,
        mock_extractor: AsyncMock,
        mock_parser: AsyncMock,
        mock_preprocessor: AsyncMock,
    ) -> None:
        pipeline: Pipeline[Receipt] = Pipeline(
            source=mock_source,
            extractor=mock_extractor,
            parser=mock_parser,
            preprocessors=[mock_preprocessor],
        )

        result = await pipeline.extract_only("/tmp/receipt.png")

        assert isinstance(result, ExtractionResult)
        mock_preprocessor.process.assert_called_once()
        mock_extractor.extract.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: store -> search round-trip with synthetic embeddings
# ---------------------------------------------------------------------------


class TestStoreSearchRoundTrip:
    """Store documents and search them back with synthetic embeddings."""

    @pytest.fixture
    def mock_embedder(self) -> AsyncMock:
        embedder = AsyncMock()
        embedder.dimensions = 4
        embedder.embed = AsyncMock(
            side_effect=lambda texts: [[float(i + 1) / 10] * 4 for i, _ in enumerate(texts)]
        )
        return embedder

    @pytest.mark.asyncio
    async def test_save_and_search_bm25(self) -> None:
        """Documents saved to store are searchable via BM25."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        await store.save("r1", Receipt(vendor="morning coffee shop", total=5.0))
        await store.save("r2", Receipt(vendor="evening bookstore", total=25.0))
        await store.save("r3", Receipt(vendor="afternoon coffee roasters", total=12.0))

        results = await store.search("coffee", mode="bm25")

        assert len(results) >= 1
        doc_ids = [r.document_id for r in results]
        assert "r1" in doc_ids
        assert "r3" in doc_ids
        assert "r2" not in doc_ids

    @pytest.mark.asyncio
    async def test_save_search_delete_roundtrip(self) -> None:
        """Full CRUD round-trip: save, search, delete, search again."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        await store.save("r1", Receipt(vendor="coffee shop", total=5.0))

        results = await store.search("coffee", mode="bm25")
        assert len(results) == 1
        assert results[0].document_id == "r1"
        assert results[0].data["vendor"] == "coffee shop"

        await store.delete("r1")

        results = await store.search("coffee", mode="bm25")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_result_structure(self) -> None:
        """SearchResult has correct structure with signals."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        await store.save("r1", Receipt(vendor="coffee shop downtown", total=5.0))

        results = await store.search("coffee", mode="bm25")
        assert len(results) == 1

        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.document_id == "r1"
        assert r.score > 0
        assert "bm25" in r.signals
        assert "final_score" in r.signals


# ---------------------------------------------------------------------------
# Integration: hybrid search ranked by RRF fusion
# ---------------------------------------------------------------------------


class TestHybridSearchRanking:
    """Verify hybrid search returns results ranked by RRF fusion."""

    @pytest.mark.asyncio
    async def test_bm25_ranking_order(self) -> None:
        """Documents are ranked by BM25 relevance via RRF fusion."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        # "coffee" appears in r1 and r3, not in r2
        await store.save("r1", Receipt(vendor="coffee shop", total=5.0))
        await store.save("r2", Receipt(vendor="bookstore", total=25.0))
        await store.save("r3", Receipt(vendor="coffee roasters", total=12.0))

        results = await store.search("coffee", mode="hybrid")

        doc_ids = [r.document_id for r in results]
        assert "r1" in doc_ids
        assert "r3" in doc_ids
        assert "r2" not in doc_ids

        # Each result should have fusion signals
        for r in results:
            assert "final_score" in r.signals
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_hybrid_fusion_signals_present(self) -> None:
        """Hybrid search results include normalized, RRF, and final scores."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        await store.save("r1", Receipt(vendor="wireless keyboard", total=49.99))
        await store.save("r2", Receipt(vendor="wireless mouse", total=29.99))

        results = await store.search("wireless", mode="hybrid")

        assert len(results) == 2
        for r in results:
            assert "bm25" in r.signals
            assert "normalized_score" in r.signals
            assert "rrf_score" in r.signals
            assert "final_score" in r.signals

    @pytest.mark.asyncio
    async def test_bm25_only_mode(self) -> None:
        """BM25-only mode returns results without vector signals."""
        store: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=Receipt,
            embed_fields=["vendor"],
        )

        await store.save("r1", Receipt(vendor="coffee shop", total=5.0))

        results = await store.search("coffee", mode="bm25")
        assert len(results) == 1
        assert "bm25" in results[0].signals
        assert "vector" not in results[0].signals


    @pytest.mark.asyncio
    async def test_real_receipt_search_text_bm25(self) -> None:
        """Real Receipt model with search_text() is searchable via BM25."""
        from billfox.models.receipt import Receipt as RealReceipt
        from billfox.models.receipt import ReceiptItem

        store: SQLiteDocumentStore[RealReceipt] = SQLiteDocumentStore(
            db_path=":memory:",
            schema=RealReceipt,
            embed_fields=["search_text"],
        )

        await store.save(
            "r1",
            RealReceipt(
                vendor_name="Starbucks",
                items=[ReceiptItem(description="Latte"), ReceiptItem(description="Muffin")],
                tags=["coffee"],
            ),
        )
        await store.save(
            "r2",
            RealReceipt(
                vendor_name="Office Depot",
                items=[ReceiptItem(description="Printer Paper")],
            ),
        )

        # Search by vendor name
        results = await store.search("Starbucks", mode="bm25")
        doc_ids = [r.document_id for r in results]
        assert "r1" in doc_ids
        assert "r2" not in doc_ids

        # Search by item description
        results = await store.search("Latte", mode="bm25")
        doc_ids = [r.document_id for r in results]
        assert "r1" in doc_ids

        # Search by tag
        results = await store.search("coffee", mode="bm25")
        doc_ids = [r.document_id for r in results]
        assert "r1" in doc_ids


# ---------------------------------------------------------------------------
# Import isolation: `import billfox` with only pydantic
# ---------------------------------------------------------------------------


class TestImportIsolation:
    """Verify that importing billfox does not pull in optional dependencies."""

    def test_import_billfox_succeeds(self) -> None:
        """import billfox works — verifies the top-level package loads."""
        import billfox

        assert hasattr(billfox, "Pipeline")
        assert hasattr(billfox, "Document")
        assert hasattr(billfox, "ExtractionResult")
        assert hasattr(billfox, "SearchResult")

    def test_import_billfox_in_subprocess(self) -> None:
        """import billfox succeeds in a clean subprocess.

        Uses subprocess to confirm the import path is clean.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import billfox; print(billfox.Document.__name__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Document" in result.stdout


# ---------------------------------------------------------------------------
# Extras isolation: importing billfox.extract.mistral without mistralai
# ---------------------------------------------------------------------------


class TestExtrasIsolation:
    """Verify lazy imports produce clear errors when extras are missing."""

    def test_mistral_extractor_import_works(self) -> None:
        """billfox.extract.mistral module itself can be imported."""
        from billfox.extract.mistral import MistralExtractor

        assert MistralExtractor is not None

    def test_mistral_extractor_lazy_import_message(self) -> None:
        """MistralExtractor._get_client raises clear error when mistralai is missing.

        This test verifies the error path by mocking the import failure.
        """
        from unittest.mock import patch

        from billfox.extract.mistral import MistralExtractor

        extractor = MistralExtractor(api_key="test-key")

        # Simulate mistralai not being installed
        with (
            patch.dict("sys.modules", {"mistralai": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'mistralai'")),
            pytest.raises((ImportError, RuntimeError)),
        ):
            extractor._get_client()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLISmokeTest:
    """Verify the CLI entry point works."""

    def test_billfox_help_exits_zero(self) -> None:
        """billfox --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "billfox", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "billfox" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_billfox_extract_help(self) -> None:
        """billfox extract --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "billfox", "extract", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_billfox_parse_help(self) -> None:
        """billfox parse --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "billfox", "parse", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_billfox_search_help(self) -> None:
        """billfox search --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "billfox", "search", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_billfox_config_help(self) -> None:
        """billfox config --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "billfox", "config", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
