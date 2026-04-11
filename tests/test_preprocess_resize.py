from __future__ import annotations

import io

import pytest
from PIL import Image

from billfox._types import Document
from billfox.preprocess.resize import ResizePreprocessor


def _make_image_bytes(w: int, h: int, fmt: str = "JPEG") -> bytes:
    """Create a minimal valid image of the given size."""
    img = Image.new("RGB", (w, h), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _image_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) of image bytes."""
    img = Image.open(io.BytesIO(data))
    return img.size


@pytest.mark.asyncio
async def test_resize_landscape_image() -> None:
    """Landscape image wider than max_side is scaled down."""
    resizer = ResizePreprocessor(max_side=100)
    doc = Document(
        content=_make_image_bytes(200, 100),
        mime_type="image/jpeg",
        source_uri="test.jpg",
    )

    result = await resizer.process(doc)

    w, h = _image_size(result.content)
    assert w == 100
    assert h == 50
    assert result.metadata["preprocessor"] == "resize"


@pytest.mark.asyncio
async def test_resize_portrait_image() -> None:
    """Portrait image taller than max_side is scaled down."""
    resizer = ResizePreprocessor(max_side=100)
    doc = Document(
        content=_make_image_bytes(50, 200),
        mime_type="image/jpeg",
        source_uri="test.jpg",
    )

    result = await resizer.process(doc)

    w, h = _image_size(result.content)
    assert w == 25
    assert h == 100


@pytest.mark.asyncio
async def test_resize_preserves_aspect_ratio() -> None:
    """Aspect ratio is preserved after resize."""
    resizer = ResizePreprocessor(max_side=500)
    doc = Document(
        content=_make_image_bytes(1000, 750),
        mime_type="image/jpeg",
        source_uri="test.jpg",
    )

    result = await resizer.process(doc)

    w, h = _image_size(result.content)
    assert w == 500
    assert h == 375


@pytest.mark.asyncio
async def test_resize_no_op_when_within_limit() -> None:
    """Image smaller than max_side is returned unchanged."""
    resizer = ResizePreprocessor(max_side=1024)
    original_bytes = _make_image_bytes(800, 600)
    doc = Document(
        content=original_bytes,
        mime_type="image/jpeg",
        source_uri="test.jpg",
    )

    result = await resizer.process(doc)

    assert result is doc  # Same object, not a copy


@pytest.mark.asyncio
async def test_resize_exact_max_side() -> None:
    """Image exactly at max_side is returned unchanged."""
    resizer = ResizePreprocessor(max_side=100)
    original_bytes = _make_image_bytes(100, 80)
    doc = Document(
        content=original_bytes,
        mime_type="image/jpeg",
        source_uri="test.jpg",
    )

    result = await resizer.process(doc)

    assert result is doc


@pytest.mark.asyncio
async def test_resize_passes_pdf_through() -> None:
    """PDF documents pass through unchanged."""
    resizer = ResizePreprocessor(max_side=100)
    doc = Document(
        content=b"%PDF-1.4 fake",
        mime_type="application/pdf",
        source_uri="test.pdf",
    )

    result = await resizer.process(doc)

    assert result is doc


@pytest.mark.asyncio
async def test_resize_png_format() -> None:
    """PNG images are resized and remain PNG."""
    resizer = ResizePreprocessor(max_side=50)
    doc = Document(
        content=_make_image_bytes(200, 100, fmt="PNG"),
        mime_type="image/png",
        source_uri="test.png",
    )

    result = await resizer.process(doc)

    w, h = _image_size(result.content)
    assert w == 50
    assert h == 25
    assert result.mime_type == "image/png"


@pytest.mark.asyncio
async def test_resize_preserves_source_uri() -> None:
    """Source URI is preserved after resize."""
    resizer = ResizePreprocessor(max_side=50)
    doc = Document(
        content=_make_image_bytes(200, 100),
        mime_type="image/jpeg",
        source_uri="/path/to/image.jpg",
    )

    result = await resizer.process(doc)

    assert result.source_uri == "/path/to/image.jpg"


@pytest.mark.asyncio
async def test_implements_preprocessor_protocol() -> None:
    """ResizePreprocessor satisfies the Preprocessor protocol."""
    from billfox.preprocess import Preprocessor

    resizer = ResizePreprocessor()
    assert isinstance(resizer, Preprocessor)
