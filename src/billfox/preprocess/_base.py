from __future__ import annotations

from typing import Protocol, runtime_checkable

from billfox._types import Document


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for preprocessing documents before extraction."""

    async def process(self, document: Document) -> Document: ...
