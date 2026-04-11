"""SQLite-backed document store with optional sqlite-vec and FTS5 support."""

from __future__ import annotations

import contextlib
import logging
import struct
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from billfox._types import SearchResult
from billfox.store._schema import Base, DocumentEmbeddingRow, DocumentRow

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SQLiteDocumentStore(Generic[T]):
    """Store and retrieve Pydantic models in SQLite.

    Optionally generates embeddings and syncs them to sqlite-vec virtual tables
    and FTS5 tables for hybrid search.
    """

    def __init__(
        self,
        *,
        db_path: str = ":memory:",
        schema: type[T],
        embedder: Any | None = None,
        embed_fields: list[str] | None = None,
    ) -> None:
        self._schema = schema
        self._embedder = embedder
        self._embed_fields = embed_fields or []
        self._sqlite_vec_available = False
        self._initialised = False

        url = f"sqlite+aiosqlite:///{db_path}" if db_path != ":memory:" else "sqlite+aiosqlite://"
        self._engine: AsyncEngine = create_async_engine(
            url,
            connect_args={"check_same_thread": False},
        )
        event.listen(self._engine.sync_engine, "connect", self._on_connect)
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _on_connect(self, dbapi_connection: Any, connection_record: Any) -> None:
        """Configure SQLite pragmas and load sqlite-vec extension."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        try:
            import sqlite_vec  # noqa: I001

            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)
            self._sqlite_vec_available = True
        except Exception:
            pass
        cursor.close()

    async def _ensure_tables(self) -> None:
        """Create core tables plus optional FTS5 and vec0 virtual tables."""
        if self._initialised:
            return

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # FTS5 virtual table for BM25 text search
            await conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings_fts "
                    "USING fts5(document_id, field_name, text_content, "
                    "content='document_embeddings', content_rowid='id')"
                )
            )

            # sqlite-vec virtual table for vector KNN search
            if self._sqlite_vec_available and self._embedder is not None:
                dims = self._embedder.dimensions
                await conn.execute(
                    text(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings_vec "
                        f"USING vec0(id TEXT PRIMARY KEY, embedding float[{dims}])"
                    )
                )

        self._initialised = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, document_id: str, data: T) -> None:
        """Save a Pydantic model instance, upserting if the ID already exists."""
        await self._ensure_tables()
        data_json = data.model_dump_json()

        async with self._session_factory() as session:
            async with session.begin():
                existing = await session.get(DocumentRow, document_id)
                if existing is not None:
                    existing.data_json = data_json
                    existing.schema_name = self._schema.__name__
                else:
                    session.add(
                        DocumentRow(
                            id=document_id,
                            schema_name=self._schema.__name__,
                            data_json=data_json,
                        )
                    )

            # Sync embed fields
            if self._embed_fields:
                await self._sync_embeddings(session, document_id, data)

    async def get(self, document_id: str) -> T | None:
        """Retrieve a document by ID, or ``None`` if not found."""
        await self._ensure_tables()
        async with self._session_factory() as session:
            row = await session.get(DocumentRow, document_id)
            if row is None:
                return None
            return self._schema.model_validate_json(row.data_json)

    async def search(self, query: str, *, limit: int = 20, mode: str = "hybrid") -> list[SearchResult]:
        """Hybrid BM25 + vector search with RRF fusion.

        *mode* controls which signals are used: ``"hybrid"`` (default),
        ``"bm25"``, or ``"vector"``.
        """
        await self._ensure_tables()

        from billfox.store._search import hybrid_search

        async with self._session_factory() as session:
            fused = await hybrid_search(
                session,
                query,
                embedder=self._embedder,
                sqlite_vec_available=self._sqlite_vec_available,
                limit=limit,
                mode=mode,
            )

            if not fused:
                return []

            # Fetch document data for results
            results: list[SearchResult] = []
            for doc_id, signals in fused:
                row = await session.get(DocumentRow, doc_id)
                if row is None:
                    continue
                data = self._schema.model_validate_json(row.data_json)
                results.append(
                    SearchResult(
                        document_id=doc_id,
                        data=data.model_dump(),
                        score=signals.get("final_score", 0.0),
                        signals=signals,
                    )
                )

            return results

    async def delete(self, document_id: str) -> None:
        """Delete a document and its associated embeddings."""
        await self._ensure_tables()
        async with self._session_factory() as session, session.begin():
            row = await session.get(DocumentRow, document_id)
            if row is not None:
                # Clean up FTS entries
                fts_del = text(
                    "INSERT INTO document_embeddings_fts("
                    "document_embeddings_fts, rowid, document_id, "
                    "field_name, text_content) "
                    "SELECT 'delete', de.id, de.document_id, "
                    "de.field_name, de.text_content "
                    "FROM document_embeddings de "
                    "WHERE de.document_id = :doc_id"
                )
                await session.execute(fts_del, {"doc_id": document_id})

                # Clean up vec entries
                if self._sqlite_vec_available and self._embedder is not None:
                    emb_rows = (
                        await session.execute(
                            select(DocumentEmbeddingRow.id).where(
                                DocumentEmbeddingRow.document_id == document_id
                            )
                        )
                    ).scalars().all()
                    for emb_id in emb_rows:
                        with contextlib.suppress(Exception):
                            await session.execute(
                                text("DELETE FROM document_embeddings_vec WHERE id = :id"),
                                {"id": f"{document_id}:{emb_id}"},
                            )

                await session.delete(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _sync_embeddings(self, session: AsyncSession, document_id: str, data: T) -> None:
        """Sync embed_fields to document_embeddings, FTS5, and vec0 tables."""
        model_dict = data.model_dump()

        async with session.begin():
            for field_name in self._embed_fields:
                value = model_dict.get(field_name)
                if value is None:
                    continue
                text_content = str(value)

                # Upsert embedding row
                existing = (
                    await session.execute(
                        select(DocumentEmbeddingRow).where(
                            DocumentEmbeddingRow.document_id == document_id,
                            DocumentEmbeddingRow.field_name == field_name,
                        )
                    )
                ).scalar_one_or_none()

                if existing is not None:
                    # Delete old FTS entry before update
                    await session.execute(
                        text(
                            "INSERT INTO document_embeddings_fts("
                            "document_embeddings_fts, rowid, document_id, "
                            "field_name, text_content) "
                            "VALUES('delete', :rowid, :doc_id, :field, :content)"
                        ),
                        {
                            "rowid": existing.id,
                            "doc_id": document_id,
                            "field": field_name,
                            "content": existing.text_content,
                        },
                    )
                    existing.text_content = text_content
                    await session.flush()
                    row_id = existing.id
                else:
                    new_row = DocumentEmbeddingRow(
                        document_id=document_id,
                        field_name=field_name,
                        text_content=text_content,
                    )
                    session.add(new_row)
                    await session.flush()
                    row_id = new_row.id

                # Insert into FTS5
                await session.execute(
                    text(
                        "INSERT INTO document_embeddings_fts(rowid, document_id, field_name, text_content) "
                        "VALUES(:rowid, :doc_id, :field, :content)"
                    ),
                    {
                        "rowid": row_id,
                        "doc_id": document_id,
                        "field": field_name,
                        "content": text_content,
                    },
                )

                # Sync vector to vec0
                if self._sqlite_vec_available and self._embedder is not None:
                    vectors = await self._embedder.embed([text_content])
                    if vectors:
                        blob = struct.pack(f"{len(vectors[0])}f", *vectors[0])
                        vec_id = f"{document_id}:{field_name}"
                        try:
                            await session.execute(
                                text(
                                    "INSERT OR REPLACE INTO document_embeddings_vec(id, embedding) "
                                    "VALUES(:id, :vec)"
                                ),
                                {"id": vec_id, "vec": blob},
                            )
                        except Exception as exc:
                            logger.debug("Could not sync vector: %s", exc)
