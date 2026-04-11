from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from billfox._types import Document, ExtractionResult
from billfox.extract._base import Extractor
from billfox.parse._base import Parser
from billfox.preprocess._base import Preprocessor
from billfox.source._base import DocumentSource
from billfox.store._base import DocumentStore

T = TypeVar("T", bound=BaseModel)


@dataclass
class Pipeline(Generic[T]):
    """Composable document processing pipeline.

    Wires together source -> preprocess -> extract -> parse -> store.
    """

    source: DocumentSource
    extractor: Extractor
    parser: Parser[T]
    preprocessors: list[Preprocessor] = field(default_factory=list)
    store: DocumentStore[T] | None = None

    async def run(self, uri: str, document_id: str | None = None) -> T:
        """Execute the full pipeline: load -> preprocess -> extract -> parse -> store."""
        document = await self.source.load(uri)
        document = await self._preprocess(document)
        result = await self.extractor.extract(document)
        parsed = await self.parser.parse(result.markdown)
        if self.store is not None and document_id is not None:
            await self.store.save(document_id, parsed)
        return parsed

    async def extract_only(self, uri: str) -> ExtractionResult:
        """Execute load -> preprocess -> extract only (no parse or store)."""
        document = await self.source.load(uri)
        document = await self._preprocess(document)
        return await self.extractor.extract(document)

    async def _preprocess(self, document: Document) -> Document:
        for preprocessor in self.preprocessors:
            document = await preprocessor.process(document)
        return document
