"""Integration tests for hybrid search — BM25 + vector + RRF fusion."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from billfox.store._search import (
    SearchCandidate,
    apply_hybrid_fusion,
    normalize_scores,
)
from billfox.store.sqlite import SQLiteDocumentStore

# --- Sample schema ---


class Product(BaseModel):
    name: str
    description: str
    price: float = 0.0


# --- Unit tests for normalize_scores ---


class TestNormalizeScores:
    def test_empty(self) -> None:
        assert normalize_scores([]) == []

    def test_single(self) -> None:
        assert normalize_scores([5.0]) == [1.0]

    def test_all_equal(self) -> None:
        assert normalize_scores([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_min_max(self) -> None:
        result = normalize_scores([0.0, 5.0, 10.0])
        assert result == [0.0, 0.5, 1.0]

    def test_negative_scores(self) -> None:
        result = normalize_scores([-10.0, 0.0, 10.0])
        assert result == [0.0, 0.5, 1.0]


# --- Unit tests for apply_hybrid_fusion ---


class TestApplyHybridFusion:
    def test_empty_candidates(self) -> None:
        assert apply_hybrid_fusion([]) == []

    def test_no_signals_sorts_by_recency(self) -> None:
        from datetime import datetime

        c1 = SearchCandidate(
            document_id="old", created_at=datetime(2020, 1, 1)
        )
        c2 = SearchCandidate(
            document_id="new", created_at=datetime(2025, 1, 1)
        )
        result = apply_hybrid_fusion([c1, c2])
        assert result[0][0] == "new"
        assert result[1][0] == "old"

    def test_single_signal(self) -> None:
        candidates = [
            SearchCandidate(document_id="a", signals={"bm25": 10.0}),
            SearchCandidate(document_id="b", signals={"bm25": 5.0}),
            SearchCandidate(document_id="c", signals={"bm25": 1.0}),
        ]
        result = apply_hybrid_fusion(candidates)
        doc_ids = [r[0] for r in result]
        assert doc_ids[0] == "a"
        assert doc_ids[-1] == "c"

        # Check signals dict contains expected keys
        signals = result[0][1]
        assert "bm25" in signals
        assert "normalized_score" in signals
        assert "rrf_score" in signals
        assert "final_score" in signals

    def test_two_signals_fused(self) -> None:
        candidates = [
            SearchCandidate(
                document_id="x", signals={"bm25": 10.0, "vector": 0.9}
            ),
            SearchCandidate(
                document_id="y", signals={"bm25": 1.0, "vector": 0.1}
            ),
        ]
        result = apply_hybrid_fusion(candidates)
        # x has the highest scores on both signals — it should rank first
        assert result[0][0] == "x"
        assert result[0][1]["final_score"] > result[1][1]["final_score"]

    def test_conflicting_signals(self) -> None:
        """When signals disagree, fusion should balance them."""
        candidates = [
            SearchCandidate(
                document_id="bm25_winner",
                signals={"bm25": 10.0, "vector": 0.1},
            ),
            SearchCandidate(
                document_id="vec_winner",
                signals={"bm25": 1.0, "vector": 0.9},
            ),
        ]
        result = apply_hybrid_fusion(candidates)
        # Both should appear, each with a non-zero final score
        scores = {r[0]: r[1]["final_score"] for r in result}
        assert scores["bm25_winner"] > 0
        assert scores["vec_winner"] > 0

    def test_rrf_constant_used(self) -> None:
        candidates = [
            SearchCandidate(document_id="a", signals={"s": 10.0}),
            SearchCandidate(document_id="b", signals={"s": 5.0}),
        ]
        result_k10 = apply_hybrid_fusion(candidates, rrf_k=10)
        result_k100 = apply_hybrid_fusion(candidates, rrf_k=100)
        # Different k values produce different RRF scores
        rrf_k10 = result_k10[0][1]["rrf_score"]
        rrf_k100 = result_k100[0][1]["rrf_score"]
        assert rrf_k10 != rrf_k100


# --- Integration tests with in-memory SQLite ---


@pytest.fixture
async def store_with_embedder() -> SQLiteDocumentStore[Product]:
    """Store with a mock embedder that returns synthetic vectors."""
    mock_embedder = AsyncMock()
    mock_embedder.dimensions = 4
    mock_embedder.embed = AsyncMock(
        side_effect=lambda texts: [[float(i + 1) / 10] * 4 for i, _ in enumerate(texts)]
    )

    return SQLiteDocumentStore(
        db_path=":memory:",
        schema=Product,
        embedder=mock_embedder,
        embed_fields=["name", "description"],
    )


@pytest.fixture
async def store_no_embedder() -> SQLiteDocumentStore[Product]:
    """Store without embedder — BM25-only search."""
    return SQLiteDocumentStore(
        db_path=":memory:",
        schema=Product,
        embed_fields=["name", "description"],
    )


async def _seed_products(store: SQLiteDocumentStore[Product]) -> None:
    """Seed store with sample products."""
    await store.save(
        "p1", Product(name="wireless keyboard", description="ergonomic bluetooth keyboard", price=49.99)
    )
    await store.save(
        "p2", Product(name="wireless mouse", description="ergonomic bluetooth mouse", price=29.99)
    )
    await store.save(
        "p3", Product(name="monitor stand", description="adjustable aluminum stand for monitors", price=39.99)
    )


async def test_bm25_search_returns_results(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """BM25-only search finds documents matching the query."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("keyboard")
    assert len(results) >= 1
    doc_ids = [r.document_id for r in results]
    assert "p1" in doc_ids


async def test_bm25_search_no_match(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """BM25 search returns empty list when nothing matches."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("xyznonexistent")
    assert results == []


async def test_bm25_search_multiple_matches(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """BM25 search returns multiple documents when query matches several."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("wireless")
    doc_ids = [r.document_id for r in results]
    assert "p1" in doc_ids
    assert "p2" in doc_ids


async def test_search_result_has_signals(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """SearchResult includes explainability signals."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("keyboard")
    assert len(results) >= 1
    r = results[0]
    assert r.score > 0
    assert "bm25" in r.signals
    assert "final_score" in r.signals


async def test_search_result_contains_data(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """SearchResult includes the document data dict."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("keyboard")
    assert len(results) >= 1
    assert results[0].data["name"] == "wireless keyboard"
    assert results[0].data["price"] == 49.99


async def test_search_respects_limit(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """Search respects the limit parameter."""
    await _seed_products(store_no_embedder)
    results = await store_no_embedder.search("ergonomic", limit=1)
    assert len(results) == 1


async def test_search_ranking_by_relevance(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """More relevant documents should rank higher."""
    await _seed_products(store_no_embedder)
    # "keyboard" should match p1 most strongly
    results = await store_no_embedder.search("keyboard")
    if len(results) > 1:
        assert results[0].document_id == "p1"


async def test_search_empty_store(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """Search on empty store returns empty list."""
    results = await store_no_embedder.search("anything")
    assert results == []


async def test_search_after_delete(store_no_embedder: SQLiteDocumentStore[Product]) -> None:
    """Deleted documents should not appear in search results."""
    await _seed_products(store_no_embedder)
    await store_no_embedder.delete("p1")
    results = await store_no_embedder.search("keyboard")
    doc_ids = [r.document_id for r in results]
    assert "p1" not in doc_ids
