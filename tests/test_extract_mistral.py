from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from billfox._types import Document, ExtractionResult, Page
from billfox.extract._base import Extractor
from billfox.extract.mistral import (
    MistralExtractor,
    _to_data_url,
    _validate_mime,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakePage:
    markdown: str


@dataclass
class FakeOCRResponse:
    pages: list[FakePage]


def _make_document(mime_type: str = "image/jpeg", content: bytes = b"fake-image") -> Document:
    return Document(content=content, mime_type=mime_type, source_uri="test.jpg")


def _make_mock_client(pages: list[FakePage] | None = None) -> MagicMock:
    """Create a mock Mistral client with a fake OCR response."""
    if pages is None:
        pages = [FakePage(markdown="# Page 1"), FakePage(markdown="## Page 2")]
    client = MagicMock()
    client.ocr.process.return_value = FakeOCRResponse(pages=pages)
    return client


# ---------------------------------------------------------------------------
# Tests: _to_data_url
# ---------------------------------------------------------------------------

class TestToDataUrl:
    def test_encodes_bytes_as_base64(self) -> None:
        result = _to_data_url(b"hello", "image/png")
        assert result == "data:image/png;base64,aGVsbG8="

    def test_pdf_mime_type(self) -> None:
        result = _to_data_url(b"\x00\x01", "application/pdf")
        assert result.startswith("data:application/pdf;base64,")


# ---------------------------------------------------------------------------
# Tests: _validate_mime
# ---------------------------------------------------------------------------

class TestValidateMime:
    @pytest.mark.parametrize(
        "mime",
        ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/heic", "application/pdf"],
    )
    def test_supported_types_pass(self, mime: str) -> None:
        _validate_mime(mime)  # should not raise

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported mime_type 'text/plain'"):
            _validate_mime("text/plain")


# ---------------------------------------------------------------------------
# Tests: MistralExtractor construction
# ---------------------------------------------------------------------------

class TestMistralExtractorInit:
    def test_raises_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Ensure MISTRAL_API_KEY is not set
            import os
            os.environ.pop("MISTRAL_API_KEY", None)
            with pytest.raises(RuntimeError, match="No Mistral API key"):
                MistralExtractor()

    def test_accepts_explicit_api_key(self) -> None:
        ext = MistralExtractor(api_key="test-key")
        assert ext._api_key == "test-key"

    def test_falls_back_to_env_var(self) -> None:
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            ext = MistralExtractor()
            assert ext._api_key == "env-key"

    def test_explicit_key_over_env(self) -> None:
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            ext = MistralExtractor(api_key="explicit-key")
            assert ext._api_key == "explicit-key"

    def test_default_model(self) -> None:
        ext = MistralExtractor(api_key="k")
        assert ext._model == "mistral-ocr-latest"

    def test_custom_model(self) -> None:
        ext = MistralExtractor(api_key="k", model="custom-model")
        assert ext._model == "custom-model"


# ---------------------------------------------------------------------------
# Tests: MistralExtractor.extract
# ---------------------------------------------------------------------------

class TestMistralExtractorExtract:
    @pytest.fixture()
    def extractor(self) -> MistralExtractor:
        return MistralExtractor(api_key="test-key")

    async def test_extract_image_returns_extraction_result(self, extractor: MistralExtractor) -> None:
        mock_client = _make_mock_client()
        with patch.object(extractor, "_get_client", return_value=mock_client):
            result = await extractor.extract(_make_document("image/jpeg"))

        assert isinstance(result, ExtractionResult)
        assert result.markdown == "# Page 1\n\n## Page 2"
        assert len(result.pages) == 2
        assert result.pages[0] == Page(index=0, markdown="# Page 1")
        assert result.pages[1] == Page(index=1, markdown="## Page 2")

    async def test_extract_image_uses_image_url(self, extractor: MistralExtractor) -> None:
        mock_client = _make_mock_client()
        with patch.object(extractor, "_get_client", return_value=mock_client):
            await extractor.extract(_make_document("image/png"))

        call_kwargs = mock_client.ocr.process.call_args
        doc_arg: dict[str, Any] = call_kwargs.kwargs.get("document") or call_kwargs[1].get("document")
        assert doc_arg["type"] == "image_url"
        assert "image_url" in doc_arg

    async def test_extract_pdf_uses_document_url(self, extractor: MistralExtractor) -> None:
        mock_client = _make_mock_client()
        doc = Document(content=b"pdf-bytes", mime_type="application/pdf", source_uri="test.pdf")
        with patch.object(extractor, "_get_client", return_value=mock_client):
            await extractor.extract(doc)

        call_kwargs = mock_client.ocr.process.call_args
        doc_arg: dict[str, Any] = call_kwargs.kwargs.get("document") or call_kwargs[1].get("document")
        assert doc_arg["type"] == "document_url"
        assert "document_url" in doc_arg

    async def test_extract_passes_model_to_client(self, extractor: MistralExtractor) -> None:
        mock_client = _make_mock_client()
        with patch.object(extractor, "_get_client", return_value=mock_client):
            await extractor.extract(_make_document())

        call_kwargs = mock_client.ocr.process.call_args
        model_arg = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert model_arg == "mistral-ocr-latest"

    async def test_extract_single_page(self, extractor: MistralExtractor) -> None:
        mock_client = _make_mock_client([FakePage(markdown="Only page")])
        with patch.object(extractor, "_get_client", return_value=mock_client):
            result = await extractor.extract(_make_document())

        assert result.markdown == "Only page"
        assert len(result.pages) == 1

    async def test_extract_unsupported_mime_raises(self, extractor: MistralExtractor) -> None:
        doc = Document(content=b"data", mime_type="text/plain", source_uri="test.txt")
        with pytest.raises(ValueError, match="Unsupported mime_type"):
            await extractor.extract(doc)


# ---------------------------------------------------------------------------
# Tests: Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_is_extractor(self) -> None:
        ext = MistralExtractor(api_key="k")
        assert isinstance(ext, Extractor)
