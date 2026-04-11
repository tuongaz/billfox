"""Hybrid search pipeline: BM25 + vector KNN with RRF fusion.

Ported from billfox-app/api/src/store/search_pipeline.py and search.py.
Generalised from expense-specific to generic document search.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    """A candidate document with ranking signals and metadata."""

    document_id: str
    signals: dict[str, float] = field(default_factory=dict)
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max normalization."""
    if not scores or len(scores) == 1:
        return [1.0] * len(scores)

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


# ---------------------------------------------------------------------------
# Hybrid fusion (60% normalised + 40% RRF)
# ---------------------------------------------------------------------------


def apply_hybrid_fusion(
    candidates: list[SearchCandidate], rrf_k: int = 60
) -> list[tuple[str, dict[str, float]]]:
    """Combine multiple ranking signals using hybrid fusion.

    1. Normalized score-based fusion (60% weight)
    2. Reciprocal Rank Fusion (40% weight)
    3. Deterministic tie-breaking by created_at recency

    Returns list of (document_id, signals_dict) sorted by final score.
    """
    if not candidates:
        return []

    # Collect all signal names
    all_signals: set[str] = set()
    for candidate in candidates:
        all_signals.update(candidate.signals.keys())

    if not all_signals:
        sorted_candidates = sorted(
            candidates, key=lambda c: c.created_at or datetime.min, reverse=True
        )
        return [(c.document_id, {"recency_only": 1.0}) for c in sorted_candidates]

    # Normalize scores per signal
    normalized_signals: dict[str, dict[str, float]] = {}

    for signal_name in all_signals:
        scores_for_signal = [
            (c.document_id, c.signals.get(signal_name, 0.0))
            for c in candidates
            if signal_name in c.signals
        ]

        if not scores_for_signal:
            continue

        doc_ids, scores = zip(*scores_for_signal, strict=True)
        normalized = normalize_scores(list(scores))
        normalized_signals[signal_name] = dict(zip(doc_ids, normalized, strict=True))

    # Normalized score-based fusion
    norm_scores: dict[str, float] = {}
    for candidate in candidates:
        score_sum = 0.0
        signal_count = 0
        for signal_name in candidate.signals:
            if (
                signal_name in normalized_signals
                and candidate.document_id in normalized_signals[signal_name]
            ):
                score_sum += normalized_signals[signal_name][candidate.document_id]
                signal_count += 1
        norm_scores[candidate.document_id] = score_sum / max(signal_count, 1)

    # RRF scores
    signal_ranks: dict[str, dict[str, int]] = {}
    for signal_name in all_signals:
        candidates_with_signal = [
            (c.document_id, c.signals.get(signal_name, 0.0))
            for c in candidates
            if signal_name in c.signals
        ]
        candidates_with_signal.sort(key=lambda x: x[1], reverse=True)
        signal_ranks[signal_name] = {
            doc_id: rank + 1
            for rank, (doc_id, _) in enumerate(candidates_with_signal)
        }

    rrf_scores: dict[str, float] = {}
    for candidate in candidates:
        rrf_score = 0.0
        for signal_name in candidate.signals:
            rank = signal_ranks[signal_name].get(candidate.document_id, 0)
            if rank > 0:
                rrf_score += 1.0 / (rrf_k + rank)
        rrf_scores[candidate.document_id] = rrf_score

    # Combine: 60% normalized score + 40% RRF score
    final_scores: list[tuple[SearchCandidate, float, dict[str, float]]] = []
    for candidate in candidates:
        norm_score = norm_scores.get(candidate.document_id, 0.0)
        rrf_score = rrf_scores.get(candidate.document_id, 0.0)
        final_score = 0.6 * norm_score + 0.4 * rrf_score

        explanation: dict[str, float] = {**candidate.signals}
        explanation["normalized_score"] = norm_score
        explanation["rrf_score"] = rrf_score
        explanation["final_score"] = final_score

        final_scores.append((candidate, final_score, explanation))

    # Sort by final score desc, then by recency for ties
    final_scores.sort(
        key=lambda x: (x[1], x[0].created_at or datetime.min), reverse=True
    )

    return [(item[0].document_id, item[2]) for item in final_scores]


# ---------------------------------------------------------------------------
# BM25 search via FTS5
# ---------------------------------------------------------------------------


async def bm25_search(
    session: AsyncSession,
    query: str,
    *,
    limit: int = 100,
) -> list[tuple[str, float]]:
    """Run BM25 full-text search via the FTS5 virtual table.

    Returns list of (document_id, bm25_score) tuples sorted by relevance.
    """
    result = await session.execute(
        text(
            "SELECT document_id, bm25(document_embeddings_fts) AS score "
            "FROM document_embeddings_fts "
            "WHERE document_embeddings_fts MATCH :query "
            "ORDER BY score "
            "LIMIT :limit"
        ),
        {"query": query, "limit": limit},
    )
    rows = result.fetchall()

    # FTS5 bm25() returns negative scores (lower = better match).
    # Negate so higher = better, matching convention used by fusion.
    return [(str(row[0]), -float(row[1])) for row in rows]


# ---------------------------------------------------------------------------
# Vector KNN search via sqlite-vec
# ---------------------------------------------------------------------------


async def vector_knn_search(
    session: AsyncSession,
    query_vector: list[float],
    *,
    k: int = 100,
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    """Perform KNN search against the document_embeddings_vec vec0 table.

    Returns list of (document_id, similarity) tuples sorted by descending
    similarity.  Similarity is ``1 - cosine_distance``.
    """
    blob = struct.pack(f"{len(query_vector)}f", *query_vector)

    try:
        result = await session.execute(
            text(
                "SELECT id, distance FROM document_embeddings_vec "
                "WHERE embedding MATCH :query AND k = :k"
            ),
            {"query": blob, "k": k},
        )
        rows = result.fetchall()
    except Exception as exc:
        logger.debug("sqlite-vec KNN search failed: %s", exc)
        return []

    results: list[tuple[str, float]] = []
    for row_id, distance in rows:
        similarity = 1.0 - float(distance)
        if similarity >= threshold:
            # row_id format is "document_id:field_name" — extract document_id
            doc_id = str(row_id).rsplit(":", 1)[0]
            results.append((doc_id, similarity))

    return results


# ---------------------------------------------------------------------------
# Hybrid search orchestrator
# ---------------------------------------------------------------------------


async def hybrid_search(
    session: AsyncSession,
    query: str,
    *,
    embedder: Any | None = None,
    sqlite_vec_available: bool = False,
    limit: int = 20,
    mode: str = "hybrid",
) -> list[tuple[str, dict[str, float]]]:
    """Run hybrid BM25 + vector search with RRF fusion.

    *mode* controls which signals are used:

    - ``"hybrid"`` (default): BM25 + vector, fused with RRF.
    - ``"bm25"``: BM25 full-text search only.
    - ``"vector"``: Vector KNN search only.

    Returns list of (document_id, signals) sorted by fused score, limited to
    *limit* results.
    """
    candidate_map: dict[str, SearchCandidate] = {}

    # --- BM25 signal ---
    if mode in ("hybrid", "bm25"):
        try:
            bm25_results = await bm25_search(session, query, limit=limit * 5)
        except Exception as exc:
            logger.debug("BM25 search failed: %s", exc)
            bm25_results = []

        for doc_id, score in bm25_results:
            if doc_id not in candidate_map:
                candidate_map[doc_id] = SearchCandidate(document_id=doc_id)
            candidate_map[doc_id].signals["bm25"] = score

    # --- Vector signal ---
    if mode in ("hybrid", "vector") and embedder is not None and sqlite_vec_available:
        try:
            vectors = await embedder.embed([query])
            if vectors:
                knn_results = await vector_knn_search(
                    session, vectors[0], k=limit * 5, threshold=0.0
                )
                for doc_id, similarity in knn_results:
                    if doc_id not in candidate_map:
                        candidate_map[doc_id] = SearchCandidate(document_id=doc_id)
                    # Keep the highest similarity per document
                    existing = candidate_map[doc_id].signals.get("vector", 0.0)
                    candidate_map[doc_id].signals["vector"] = max(existing, similarity)
        except Exception as exc:
            logger.debug("Vector search failed: %s", exc)

    if not candidate_map:
        return []

    # Fetch created_at for tie-breaking
    doc_ids = list(candidate_map.keys())
    try:
        placeholders = ", ".join(f":id{i}" for i in range(len(doc_ids)))
        params = {f"id{i}": did for i, did in enumerate(doc_ids)}
        result = await session.execute(
            text(
                f"SELECT id, created_at FROM documents WHERE id IN ({placeholders})"  # noqa: S608
            ),
            params,
        )
        for row in result.fetchall():
            doc_id = str(row[0])
            if doc_id in candidate_map and row[1] is not None:
                candidate_map[doc_id].created_at = row[1]
    except Exception as exc:
        logger.debug("Could not fetch created_at for tie-breaking: %s", exc)

    # Fuse and rank
    fused = apply_hybrid_fusion(list(candidate_map.values()))
    return fused[:limit]
