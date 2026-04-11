"""Integration tests for SQLiteDocumentStore — save, get, delete round-trip."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from billfox.store.sqlite import SQLiteDocumentStore

# --- Sample schema ---


class Invoice(BaseModel):
    vendor_name: str
    total: float
    currency: str = "USD"


# --- Fixtures ---


@pytest.fixture
async def store() -> SQLiteDocumentStore[Invoice]:
    return SQLiteDocumentStore(db_path=":memory:", schema=Invoice)


# --- Tests ---


async def test_save_and_get(store: SQLiteDocumentStore[Invoice]) -> None:
    inv = Invoice(vendor_name="Acme Corp", total=99.99)
    await store.save("doc-1", inv)

    result = await store.get("doc-1")
    assert result is not None
    assert result.vendor_name == "Acme Corp"
    assert result.total == 99.99
    assert result.currency == "USD"


async def test_get_missing_returns_none(store: SQLiteDocumentStore[Invoice]) -> None:
    result = await store.get("nonexistent")
    assert result is None


async def test_save_overwrite(store: SQLiteDocumentStore[Invoice]) -> None:
    inv1 = Invoice(vendor_name="Acme Corp", total=50.0)
    await store.save("doc-1", inv1)

    inv2 = Invoice(vendor_name="Acme Corp", total=75.0, currency="EUR")
    await store.save("doc-1", inv2)

    result = await store.get("doc-1")
    assert result is not None
    assert result.total == 75.0
    assert result.currency == "EUR"


async def test_delete(store: SQLiteDocumentStore[Invoice]) -> None:
    inv = Invoice(vendor_name="Delete Me", total=1.0)
    await store.save("doc-del", inv)

    await store.delete("doc-del")
    result = await store.get("doc-del")
    assert result is None


async def test_delete_nonexistent_is_noop(store: SQLiteDocumentStore[Invoice]) -> None:
    # Should not raise
    await store.delete("no-such-doc")


async def test_multiple_documents(store: SQLiteDocumentStore[Invoice]) -> None:
    await store.save("a", Invoice(vendor_name="A", total=1.0))
    await store.save("b", Invoice(vendor_name="B", total=2.0))
    await store.save("c", Invoice(vendor_name="C", total=3.0))

    a = await store.get("a")
    b = await store.get("b")
    c = await store.get("c")

    assert a is not None and a.vendor_name == "A"
    assert b is not None and b.vendor_name == "B"
    assert c is not None and c.vendor_name == "C"


async def test_search_returns_empty_list(store: SQLiteDocumentStore[Invoice]) -> None:
    # Search is a placeholder until US-010
    results = await store.search("anything")
    assert results == []


async def test_save_with_embed_fields() -> None:
    """Store with embed_fields but no embedder still works (skips embedding)."""
    s: SQLiteDocumentStore[Invoice] = SQLiteDocumentStore(
        db_path=":memory:",
        schema=Invoice,
        embed_fields=["vendor_name"],
    )
    inv = Invoice(vendor_name="Test", total=10.0)
    await s.save("doc-emb", inv)

    result = await s.get("doc-emb")
    assert result is not None
    assert result.vendor_name == "Test"


async def test_schema_name_stored() -> None:
    """Verify that schema_name column stores the Pydantic class name."""
    from sqlalchemy import select as sa_select

    from billfox.store._schema import DocumentRow

    s: SQLiteDocumentStore[Invoice] = SQLiteDocumentStore(
        db_path=":memory:", schema=Invoice
    )
    await s.save("doc-schema", Invoice(vendor_name="X", total=0))

    async with s._session_factory() as session:
        row = (
            await session.execute(
                sa_select(DocumentRow).where(DocumentRow.id == "doc-schema")
            )
        ).scalar_one()
        assert row.schema_name == "Invoice"


async def test_delete_cascades_embeddings() -> None:
    """Verify delete removes related embedding rows."""
    from sqlalchemy import func
    from sqlalchemy import select as sa_select

    from billfox.store._schema import DocumentEmbeddingRow

    s: SQLiteDocumentStore[Invoice] = SQLiteDocumentStore(
        db_path=":memory:",
        schema=Invoice,
        embed_fields=["vendor_name"],
    )
    inv = Invoice(vendor_name="Cascade Test", total=5.0)
    await s.save("doc-cascade", inv)

    # Verify embedding row exists
    async with s._session_factory() as session:
        count_before = (
            await session.execute(
                sa_select(func.count()).where(
                    DocumentEmbeddingRow.document_id == "doc-cascade"
                )
            )
        ).scalar_one()
        assert count_before >= 1

    await s.delete("doc-cascade")

    async with s._session_factory() as session:
        count_after = (
            await session.execute(
                sa_select(func.count()).where(
                    DocumentEmbeddingRow.document_id == "doc-cascade"
                )
            )
        ).scalar_one()
        assert count_after == 0
