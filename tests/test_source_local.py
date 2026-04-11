from __future__ import annotations

import pytest

from billfox._types import Document
from billfox.source.local import LocalFileSource


@pytest.fixture
def source() -> LocalFileSource:
    return LocalFileSource()


@pytest.mark.asyncio
async def test_load_jpeg(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    img = tmp / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    doc = await source.load(str(img))

    assert isinstance(doc, Document)
    assert doc.content == b"\xff\xd8\xff\xe0fake-jpeg"
    assert doc.mime_type == "image/jpeg"
    assert doc.source_uri == str(img)
    assert doc.metadata == {}


@pytest.mark.asyncio
async def test_load_png(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    img = tmp / "image.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")

    doc = await source.load(str(img))

    assert doc.mime_type == "image/png"
    assert doc.content == b"\x89PNG\r\n\x1a\nfake-png"


@pytest.mark.asyncio
async def test_load_pdf(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    pdf = tmp / "document.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake-pdf")

    doc = await source.load(str(pdf))

    assert doc.mime_type == "application/pdf"
    assert doc.content == b"%PDF-1.4 fake-pdf"


@pytest.mark.asyncio
async def test_load_webp(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    img = tmp / "image.webp"
    img.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

    doc = await source.load(str(img))

    assert doc.mime_type == "image/webp"


@pytest.mark.asyncio
async def test_load_heic(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    img = tmp / "image.heic"
    img.write_bytes(b"fake-heic")

    doc = await source.load(str(img))

    assert doc.mime_type == "image/heic"


@pytest.mark.asyncio
async def test_load_tiff(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    img = tmp / "image.tiff"
    img.write_bytes(b"II*\x00fake-tiff")

    doc = await source.load(str(img))

    assert doc.mime_type == "image/tiff"


@pytest.mark.asyncio
async def test_missing_file(source: LocalFileSource) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        await source.load("/nonexistent/path/file.jpg")


@pytest.mark.asyncio
async def test_unsupported_extension(tmp_path: object, source: LocalFileSource) -> None:
    from pathlib import Path

    tmp = Path(str(tmp_path))
    txt = tmp / "notes.txt"
    txt.write_bytes(b"hello")

    with pytest.raises(ValueError, match="Unsupported file type"):
        await source.load(str(txt))


@pytest.mark.asyncio
async def test_implements_document_source_protocol(source: LocalFileSource) -> None:
    from billfox.source import DocumentSource

    assert isinstance(source, DocumentSource)
