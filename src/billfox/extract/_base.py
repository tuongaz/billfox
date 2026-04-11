from __future__ import annotations

from typing import Protocol, runtime_checkable

from billfox._types import Document, ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Protocol for extracting text/markdown from documents."""

    async def extract(self, document: Document) -> ExtractionResult: ...
