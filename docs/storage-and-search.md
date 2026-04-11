# Storage and Search

billfox includes a SQLite-backed document store with hybrid search combining BM25 full-text and vector similarity.

## Setup

```bash
pip install 'billfox[store]'
# For vector search, also install the openai extra:
pip install 'billfox[store,openai]'
```

## Storing Documents

```python
import asyncio
from pydantic import BaseModel
from billfox.store import SQLiteDocumentStore

class Invoice(BaseModel):
    vendor_name: str
    total: float
    date: str

async def main():
    store = SQLiteDocumentStore(
        db_path="invoices.db",
        schema=Invoice,
    )

    invoice = Invoice(vendor_name="Acme Corp", total=99.50, date="2025-01-15")
    await store.save("inv-001", invoice)

    # Retrieve
    result = await store.get("inv-001")
    print(result.vendor_name)  # "Acme Corp"

    # Delete
    await store.delete("inv-001")

asyncio.run(main())
```

## Hybrid Search

To enable hybrid search (BM25 + vector), provide an embedder and specify which fields to embed:

```python
from billfox.embed import OpenAIEmbedder
from billfox.store import SQLiteDocumentStore

store = SQLiteDocumentStore(
    db_path="invoices.db",
    schema=Invoice,
    embedder=OpenAIEmbedder(),  # uses OPENAI_API_KEY env var
    embed_fields=["vendor_name"],
)

# Save some documents (embeddings are generated automatically)
await store.save("inv-001", Invoice(vendor_name="Coffee House", total=5.50, date="2025-01-15"))
await store.save("inv-002", Invoice(vendor_name="Tech Store", total=299.00, date="2025-01-16"))

# Search
results = await store.search("coffee")
for r in results:
    print(f"{r.document_id}: {r.score:.4f}")
    print(f"  signals: {r.signals}")
```

## Search Modes

The `search()` method supports three modes:

```python
# Hybrid (default): BM25 + vector with RRF fusion
results = await store.search("query", mode="hybrid")

# BM25 only: full-text search via FTS5
results = await store.search("query", mode="bm25")

# Vector only: cosine similarity via sqlite-vec
results = await store.search("query", mode="vector")
```

## How Hybrid Search Works

1. **BM25 signal** -- FTS5 full-text search scores documents by term relevance
2. **Vector signal** -- sqlite-vec computes cosine similarity against embedded fields
3. **Normalization** -- each signal is min-max normalized to [0, 1]
4. **RRF fusion** -- 60% normalized score + 40% Reciprocal Rank Fusion (k=60)
5. **Tie-breaking** -- documents with the same final score are ordered by recency

Each `SearchResult` includes a `signals` dict with the individual scores for explainability:

```python
{
    "bm25": 2.31,
    "vector": 0.87,
    "normalized_score": 0.92,
    "rrf_score": 0.016,
    "final_score": 0.558,
}
```

## Using with Pipeline

Pass the store to a `Pipeline` to automatically save parsed results:

```python
from billfox import Pipeline

pipeline = Pipeline(
    source=source,
    extractor=extractor,
    parser=parser,
    store=store,
)

# document_id must be provided to trigger storage
result = await pipeline.run("invoice.pdf", document_id="inv-001")
```

## The DocumentStore Protocol

You can implement your own store backend:

```python
from billfox.store._base import DocumentStore

class DocumentStore(Protocol[T]):
    async def save(self, document_id: str, data: T) -> None: ...
    async def get(self, document_id: str) -> T | None: ...
    async def search(self, query: str, *, limit: int = 20) -> list[SearchResult]: ...
    async def delete(self, document_id: str) -> None: ...
```
