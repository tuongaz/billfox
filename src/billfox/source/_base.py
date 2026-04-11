from __future__ import annotations

from typing import Protocol, runtime_checkable

from billfox._types import Document


@runtime_checkable
class DocumentSource(Protocol):
    """Protocol for loading documents from a source."""

    async def load(self, uri: str) -> Document: ...
