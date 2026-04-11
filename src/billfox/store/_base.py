from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from billfox._types import SearchResult

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class DocumentStore(Protocol[T]):
    """Protocol for storing and searching documents."""

    async def save(self, document_id: str, data: T) -> None: ...

    async def get(self, document_id: str) -> T | None: ...

    async def search(self, query: str, *, limit: int = 20) -> list[SearchResult]: ...

    async def delete(self, document_id: str) -> None: ...
