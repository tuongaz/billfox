from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

from billfox._types import Document, ExtractionResult, Page
from billfox.extract._base import StepCallback

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/heic",
    "application/pdf",
})


def _to_data_url(content: bytes, mime_type: str) -> str:
    """Convert raw bytes to a base64 data URL."""
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"


def _validate_mime(mime_type: str) -> None:
    """Raise ValueError if the MIME type is not supported by Mistral OCR."""
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported mime_type '{mime_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )


class MistralExtractor:
    """Extract text/markdown from documents using Mistral OCR API.

    Implements the Extractor protocol.

    Args:
        api_key: Mistral API key. Falls back to MISTRAL_API_KEY env var.
        model: Mistral OCR model name. Defaults to ``mistral-ocr-latest``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "mistral-ocr-latest",
    ) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError(
                "No Mistral API key provided. Pass api_key= or set the "
                "MISTRAL_API_KEY environment variable."
            )
        self._api_key = key
        self._model = model

    def _get_client(self) -> Any:
        """Lazily import and return a Mistral client."""
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai is required for MistralExtractor. "
                "Install it with: pip install 'billfox[mistral]'"
            ) from None
        return Mistral(api_key=self._api_key)

    def _ocr_sync(
        self,
        document: Document,
        on_step: StepCallback | None,
    ) -> ExtractionResult:
        """Run OCR synchronously (called via asyncio.to_thread)."""
        _validate_mime(document.mime_type)

        def step(msg: str) -> None:
            if on_step is not None:
                on_step(msg)

        step("connecting to Mistral API")
        client = self._get_client()

        data_url = _to_data_url(document.content, document.mime_type)

        # Mistral OCR uses image_url for images, document_url for PDFs
        if document.mime_type.startswith("image/"):
            doc: dict[str, str] = {"type": "image_url", "image_url": data_url}
        else:
            doc = {"type": "document_url", "document_url": data_url}

        step("running OCR")
        resp = client.ocr.process(
            model=self._model,
            document=doc,
            include_image_base64=True,
        )

        step("extracting pages")
        pages = [
            Page(index=i, markdown=p.markdown)
            for i, p in enumerate(resp.pages)
        ]
        markdown = "\n\n".join(p.markdown for p in pages)

        return ExtractionResult(markdown=markdown, pages=pages)

    async def extract(
        self,
        document: Document,
        *,
        on_step: StepCallback | None = None,
    ) -> ExtractionResult:
        """Extract markdown from a document using Mistral OCR.

        Args:
            document: The document to extract text from.
            on_step: Optional callback for sub-step progress messages.

        Returns:
            ExtractionResult with markdown content and pages.
        """
        return await asyncio.to_thread(self._ocr_sync, document, on_step)
