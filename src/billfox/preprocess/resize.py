"""Image resize preprocessor that maintains aspect ratio."""

from __future__ import annotations

import io

from billfox._types import Document

_IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/heic",
})


class ResizePreprocessor:
    """Resize images so the longest side does not exceed max_side.

    Maintains aspect ratio. Only processes image MIME types;
    PDFs and other documents pass through unchanged.
    """

    def __init__(self, max_side: int = 1024) -> None:
        self._max_side = max_side

    async def process(self, document: Document) -> Document:
        """Resize the document image if it exceeds max_side."""
        if document.mime_type not in _IMAGE_MIME_TYPES:
            return document

        from PIL import Image

        img = Image.open(io.BytesIO(document.content))
        w, h = img.size
        m = max(w, h)

        if m <= self._max_side:
            return document

        scale = self._max_side / float(m)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BILINEAR
        resized = img.resize((new_w, new_h), resample)

        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        resized.save(buf, format=fmt)

        return Document(
            content=buf.getvalue(),
            mime_type=document.mime_type,
            source_uri=document.source_uri,
            metadata={**document.metadata, "preprocessor": "resize"},
        )
