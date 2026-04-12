from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from billfox._types import Document, ExtractionResult

StepCallback = Callable[[str], None]


@runtime_checkable
class Extractor(Protocol):
    """Protocol for extracting text/markdown from documents."""

    async def extract(
        self,
        document: Document,
        *,
        on_step: StepCallback | None = None,
    ) -> ExtractionResult: ...
