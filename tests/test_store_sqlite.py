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


async def test_list_documents_empty(store: SQLiteDocumentStore[Invoice]) -> None:
    items, total = await store.list_documents()
    assert items == []
    assert total == 0


async def test_list_documents_returns_all(store: SQLiteDocumentStore[Invoice]) -> None:
    await store.save("a", Invoice(vendor_name="A", total=1.0))
    await store.save("b", Invoice(vendor_name="B", total=2.0))

    items, total = await store.list_documents()
    assert total == 2
    assert len(items) == 2
    doc_ids = [doc_id for doc_id, _ in items]
    assert "a" in doc_ids
    assert "b" in doc_ids


async def test_list_documents_pagination(store: SQLiteDocumentStore[Invoice]) -> None:
    for i in range(5):
        await store.save(f"doc-{i}", Invoice(vendor_name=f"V{i}", total=float(i)))

    items_p1, total = await store.list_documents(limit=2, offset=0)
    assert total == 5
    assert len(items_p1) == 2

    items_p2, _ = await store.list_documents(limit=2, offset=2)
    assert len(items_p2) == 2

    items_p3, _ = await store.list_documents(limit=2, offset=4)
    assert len(items_p3) == 1

    all_ids = [doc_id for doc_id, _ in items_p1 + items_p2 + items_p3]
    assert len(set(all_ids)) == 5


async def test_list_documents_sort_created_at_asc(store: SQLiteDocumentStore[Invoice]) -> None:
    await store.save("a", Invoice(vendor_name="A", total=1.0))
    await store.save("b", Invoice(vendor_name="B", total=2.0))
    await store.save("c", Invoice(vendor_name="C", total=3.0))

    items, _ = await store.list_documents(sort="created_at", direction="asc")
    ids = [doc_id for doc_id, _ in items]
    assert ids == ["a", "b", "c"]


async def test_list_documents_sort_created_at_desc(store: SQLiteDocumentStore[Invoice]) -> None:
    await store.save("a", Invoice(vendor_name="A", total=1.0))
    await store.save("b", Invoice(vendor_name="B", total=2.0))
    await store.save("c", Invoice(vendor_name="C", total=3.0))

    items, _ = await store.list_documents(sort="created_at", direction="desc")
    ids = [doc_id for doc_id, _ in items]
    assert ids == ["c", "b", "a"]


async def test_list_documents_sort_expense_date() -> None:
    from billfox.models.receipt import Receipt

    store_r: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
        db_path=":memory:", schema=Receipt,
    )
    await store_r.save("old", Receipt(expense_date="2024-01-01T00:00:00", vendor_name="Old"))
    await store_r.save("new", Receipt(expense_date="2024-06-15T00:00:00", vendor_name="New"))
    await store_r.save("mid", Receipt(expense_date="2024-03-10T00:00:00", vendor_name="Mid"))

    items_desc, _ = await store_r.list_documents(sort="expense_date", direction="desc")
    ids_desc = [doc_id for doc_id, _ in items_desc]
    assert ids_desc == ["new", "mid", "old"]

    items_asc, _ = await store_r.list_documents(sort="expense_date", direction="asc")
    ids_asc = [doc_id for doc_id, _ in items_asc]
    assert ids_asc == ["old", "mid", "new"]


async def test_list_documents_sort_expense_date_nulls_last() -> None:
    from billfox.models.receipt import Receipt

    store_r: SQLiteDocumentStore[Receipt] = SQLiteDocumentStore(
        db_path=":memory:", schema=Receipt,
    )
    await store_r.save("nodate", Receipt(vendor_name="NoDate"))
    await store_r.save("dated", Receipt(expense_date="2024-06-15T00:00:00", vendor_name="Dated"))

    items_desc, _ = await store_r.list_documents(sort="expense_date", direction="desc")
    ids = [doc_id for doc_id, _ in items_desc]
    assert ids[-1] == "nodate"

    items_asc, _ = await store_r.list_documents(sort="expense_date", direction="asc")
    ids_asc = [doc_id for doc_id, _ in items_asc]
    assert ids_asc[-1] == "nodate"


async def test_save_and_get_file_paths(store: SQLiteDocumentStore[Invoice]) -> None:
    inv = Invoice(vendor_name="Acme", total=10.0)
    await store.save("doc-fp", inv)

    await store.save_file_paths(
        "doc-fp",
        file_path="/backups/2025/06/15/receipt.jpg",
        original_file_path="/backups/2025/06/15/receipt_original.jpg",
    )

    fp, ofp = await store.get_file_paths("doc-fp")
    assert fp == "/backups/2025/06/15/receipt.jpg"
    assert ofp == "/backups/2025/06/15/receipt_original.jpg"


async def test_get_file_paths_missing_doc(store: SQLiteDocumentStore[Invoice]) -> None:
    fp, ofp = await store.get_file_paths("nonexistent")
    assert fp is None
    assert ofp is None


async def test_get_file_paths_no_paths_stored(store: SQLiteDocumentStore[Invoice]) -> None:
    inv = Invoice(vendor_name="X", total=1.0)
    await store.save("doc-no-fp", inv)

    fp, ofp = await store.get_file_paths("doc-no-fp")
    assert fp is None
    assert ofp is None


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


class ModelWithSearchText(BaseModel):
    """Test model with a search_text() method for callable embed_fields."""
    name: str
    category: str = ""

    def search_text(self) -> str:
        parts = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.category:
            parts.append(f"Category: {self.category}")
        return "\n".join(parts)


async def test_callable_embed_field() -> None:
    """_sync_embeddings calls a method when embed_fields names a callable."""
    from sqlalchemy import select as sa_select

    from billfox.store._schema import DocumentEmbeddingRow

    s: SQLiteDocumentStore[ModelWithSearchText] = SQLiteDocumentStore(
        db_path=":memory:",
        schema=ModelWithSearchText,
        embed_fields=["search_text"],
    )
    await s.save("doc-callable", ModelWithSearchText(name="Coffee Shop", category="food"))

    async with s._session_factory() as session:
        row = (
            await session.execute(
                sa_select(DocumentEmbeddingRow).where(
                    DocumentEmbeddingRow.document_id == "doc-callable",
                    DocumentEmbeddingRow.field_name == "search_text",
                )
            )
        ).scalar_one()
        assert "Name: Coffee Shop" in row.text_content
        assert "Category: food" in row.text_content


async def test_callable_embed_field_bm25_search() -> None:
    """Callable embed_fields content is searchable via BM25."""
    s: SQLiteDocumentStore[ModelWithSearchText] = SQLiteDocumentStore(
        db_path=":memory:",
        schema=ModelWithSearchText,
        embed_fields=["search_text"],
    )
    await s.save("d1", ModelWithSearchText(name="morning coffee shop", category="cafe"))
    await s.save("d2", ModelWithSearchText(name="evening bookstore", category="books"))

    results = await s.search("coffee", mode="bm25")
    doc_ids = [r.document_id for r in results]
    assert "d1" in doc_ids
    assert "d2" not in doc_ids
