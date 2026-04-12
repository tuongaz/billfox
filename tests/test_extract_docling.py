from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from billfox._types import Document, ExtractionResult, Page
from billfox.extract._base import Extractor
from billfox.extract.docling import (
    DoclingExtractor,
    _validate_mime,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeDoclingDocument:
    pages: dict[int, object]
    _full_markdown: str
    _page_markdowns: dict[int, str] = field(default_factory=dict)

    def export_to_markdown(self, **kwargs: Any) -> str:
        page_no = kwargs.get("page_no")
        if page_no is None:
            return self._full_markdown
        return self._page_markdowns[page_no]


@dataclass
class FakeConversionResult:
    document: FakeDoclingDocument


def _make_document(mime_type: str = "image/jpeg", content: bytes = b"fake-image") -> Document:
    return Document(content=content, mime_type=mime_type, source_uri="test.jpg")


def _make_fake_result(
    pages: dict[int, str] | None = None,
    full_markdown: str | None = None,
) -> FakeConversionResult:
    """Create a fake Docling ConversionResult."""
    if pages is None:
        pages = {1: "# Page 1", 2: "## Page 2"}
    if full_markdown is None:
        full_markdown = "\n\n".join(pages[k] for k in sorted(pages))
    doc = FakeDoclingDocument(
        pages={k: object() for k in pages},
        _full_markdown=full_markdown,
        _page_markdowns=pages,
    )
    return FakeConversionResult(document=doc)


# ---------------------------------------------------------------------------
# Tests: _validate_mime
# ---------------------------------------------------------------------------


class TestValidateMime:
    @pytest.mark.parametrize(
        "mime",
        ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp", "application/pdf"],
    )
    def test_supported_types_pass(self, mime: str) -> None:
        _validate_mime(mime)  # should not raise

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported mime_type 'text/plain'"):
            _validate_mime("text/plain")

    def test_heic_not_supported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported mime_type 'image/heic'"):
            _validate_mime("image/heic")


# ---------------------------------------------------------------------------
# Tests: DoclingExtractor construction
# ---------------------------------------------------------------------------


class TestDoclingExtractorInit:
    def test_creates_without_args(self) -> None:
        ext = DoclingExtractor()
        assert ext._converter is None

    def test_converter_initially_none(self) -> None:
        ext = DoclingExtractor()
        assert ext._converter is None


# ---------------------------------------------------------------------------
# Tests: DoclingExtractor.extract
# ---------------------------------------------------------------------------


class TestDoclingExtractorExtract:
    @pytest.fixture()
    def extractor(self) -> DoclingExtractor:
        return DoclingExtractor()

    async def test_extract_returns_extraction_result(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result()
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        with patch.object(extractor, "_get_converter", return_value=mock_converter):
            result = await extractor.extract(_make_document("image/jpeg"))

        assert isinstance(result, ExtractionResult)
        assert result.markdown == "# Page 1\n\n## Page 2"
        assert len(result.pages) == 2
        assert result.pages[0] == Page(index=0, markdown="# Page 1")
        assert result.pages[1] == Page(index=1, markdown="## Page 2")

    async def test_extract_pdf(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result({1: "PDF content"}, "PDF content")
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        doc = Document(content=b"pdf-bytes", mime_type="application/pdf", source_uri="test.pdf")
        with patch.object(extractor, "_get_converter", return_value=mock_converter):
            result = await extractor.extract(doc)

        assert isinstance(result, ExtractionResult)
        assert result.markdown == "PDF content"
        assert len(result.pages) == 1

    async def test_extract_passes_document_stream(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result({1: "text"}, "text")
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        with (
            patch.object(extractor, "_get_converter", return_value=mock_converter),
            patch("docling.datamodel.base_models.DocumentStream") as mock_stream_cls,
        ):
            await extractor.extract(_make_document("image/png"))

        mock_stream_cls.assert_called_once()
        call_kwargs = mock_stream_cls.call_args
        assert call_kwargs.kwargs.get("name") == "document.png" or call_kwargs[1].get("name") == "document.png"

    async def test_extract_single_page(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result({1: "Only page"}, "Only page")
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        with patch.object(extractor, "_get_converter", return_value=mock_converter):
            result = await extractor.extract(_make_document())

        assert result.markdown == "Only page"
        assert len(result.pages) == 1
        assert result.pages[0] == Page(index=0, markdown="Only page")

    async def test_extract_page_indices_are_zero_based(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result({1: "P1", 2: "P2", 3: "P3"}, "P1\n\nP2\n\nP3")
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        with patch.object(extractor, "_get_converter", return_value=mock_converter):
            result = await extractor.extract(_make_document())

        assert result.pages[0].index == 0
        assert result.pages[1].index == 1
        assert result.pages[2].index == 2

    async def test_extract_unsupported_mime_raises(self, extractor: DoclingExtractor) -> None:
        doc = Document(content=b"data", mime_type="text/plain", source_uri="test.txt")
        with pytest.raises(ValueError, match="Unsupported mime_type"):
            await extractor.extract(doc)

    async def test_converter_is_cached(self, extractor: DoclingExtractor) -> None:
        fake_result = _make_fake_result({1: "text"}, "text")
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_result

        # First call sets the converter
        with patch.object(extractor, "_get_converter", return_value=mock_converter) as mock_get:
            await extractor.extract(_make_document())
            await extractor.extract(_make_document())

        assert mock_get.call_count == 2

        # Verify real caching behavior: once _converter is set, _get_converter returns it
        ext2 = DoclingExtractor()
        ext2._converter = mock_converter
        assert ext2._get_converter() is mock_converter


# ---------------------------------------------------------------------------
# Tests: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_is_extractor(self) -> None:
        ext = DoclingExtractor()
        assert isinstance(ext, Extractor)
