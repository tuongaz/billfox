from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

from billfox._types import Document, ExtractionResult, Page
from billfox.extract._base import StepCallback

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "application/pdf",
})

_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}


def _validate_mime(mime_type: str) -> None:
    """Raise ValueError if the MIME type is not supported by Docling."""
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported mime_type '{mime_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )


class DoclingExtractor:
    """Extract text/markdown from documents using Docling.

    Implements the Extractor protocol.

    Docling runs locally — no API key is required.
    """

    def __init__(self) -> None:
        self._converter: Any | None = None

    def _get_converter(self) -> Any:
        """Lazily import and return a Docling DocumentConverter."""
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError(
                "docling is required for DoclingExtractor. "
                f"Install it with: pip install 'billfox[docling]' ({exc})"
            ) from None
        self._converter = DocumentConverter()
        return self._converter

    def _convert_sync(
        self,
        document: Document,
        on_step: StepCallback | None,
    ) -> ExtractionResult:
        """Run conversion synchronously (called via asyncio.to_thread)."""
        _validate_mime(document.mime_type)

        def step(msg: str) -> None:
            if on_step is not None:
                on_step(msg)

        step("initializing OCR model")
        converter = self._get_converter()

        from docling.datamodel.base_models import DocumentStream

        ext = _MIME_TO_EXT.get(document.mime_type, ".bin")
        stream = DocumentStream(
            name=f"document{ext}",
            stream=BytesIO(document.content),
        )

        step("converting document")
        result = converter.convert(stream)

        step("extracting pages")
        pages: list[Page] = []
        for page_no in sorted(result.document.pages.keys()):
            page_md = result.document.export_to_markdown(page_no=page_no)
            pages.append(Page(index=page_no - 1, markdown=page_md))

        markdown = result.document.export_to_markdown()

        return ExtractionResult(markdown=markdown, pages=pages)

    async def extract(
        self,
        document: Document,
        *,
        on_step: StepCallback | None = None,
    ) -> ExtractionResult:
        """Extract markdown from a document using Docling.

        Args:
            document: The document to extract text from.
            on_step: Optional callback for sub-step progress messages.

        Returns:
            ExtractionResult with markdown content and pages.
        """
        return await asyncio.to_thread(self._convert_sync, document, on_step)
