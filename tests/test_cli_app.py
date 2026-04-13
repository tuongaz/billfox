"""Tests for billfox CLI extract command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from billfox._types import ExtractionResult, Page
from billfox.cli.app import app

runner = CliRunner()


# ── extract command ──────────────────────────────────────────────


class TestExtractCommand:
    """Tests for the extract subcommand."""

    def test_extract_outputs_markdown_to_stdout(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/jpeg")
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                markdown="# Hello\nExtracted text",
                pages=[Page(index=0, markdown="# Hello\nExtracted text")],
            )
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch("billfox.extract.docling.DoclingExtractor", return_value=mock_extractor),
        ):
            result = runner.invoke(app, ["extract", str(img), "--extractor", "docling"])

        assert result.exit_code == 0
        assert "# Hello" in result.output
        assert "Extracted text" in result.output

    def test_extract_writes_to_output_file(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNGfake")
        out = tmp_path / "output.md"

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/png")
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                markdown="Extracted markdown",
                pages=[Page(index=0, markdown="Extracted markdown")],
            )
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch("billfox.extract.docling.DoclingExtractor", return_value=mock_extractor),
        ):
            result = runner.invoke(app, ["extract", str(img), "--extractor", "docling", "--output", str(out)])

        assert result.exit_code == 0
        assert out.read_text() == "Extracted markdown"

    def test_extract_missing_file(self) -> None:
        mock_source = MagicMock()
        mock_source.load = AsyncMock(side_effect=FileNotFoundError("File not found: /nonexistent"))

        with patch("billfox.source.local.LocalFileSource", return_value=mock_source):
            result = runner.invoke(app, ["extract", "/nonexistent.jpg", "--extractor", "docling"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_extract_unknown_extractor(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        result = runner.invoke(app, ["extract", str(img), "--extractor", "unknown"])
        assert result.exit_code != 0

    def test_extract_with_preprocess_resize(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/jpeg")
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(markdown="text", pages=[])
        )
        mock_resize = MagicMock()
        mock_resize.process = AsyncMock(
            return_value=MagicMock(content=b"resized", mime_type="image/jpeg")
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch("billfox.extract.docling.DoclingExtractor", return_value=mock_extractor),
            patch("billfox.preprocess.resize.ResizePreprocessor", return_value=mock_resize),
        ):
            result = runner.invoke(app, ["extract", str(img), "--extractor", "docling", "--preprocess", "resize"])

        assert result.exit_code == 0
        mock_resize.process.assert_awaited_once()

    def test_extract_with_api_key(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/jpeg")
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(markdown="text", pages=[])
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch("billfox.extract.mistral.MistralExtractor", return_value=mock_extractor) as mock_cls,
        ):
            result = runner.invoke(app, ["extract", str(img), "--extractor", "mistral", "--api-key", "test-key"])

        assert result.exit_code == 0
        mock_cls.assert_called_once_with(api_key="test-key")

    def test_extract_verbose(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/jpeg")
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(markdown="text", pages=[])
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch("billfox.extract.docling.DoclingExtractor", return_value=mock_extractor),
        ):
            result = runner.invoke(app, ["extract", str(img), "--extractor", "docling", "--verbose"])

        assert result.exit_code == 0


# ── help ─────────────────────────────────────────────────────────


class TestHelp:
    """Tests for CLI help output."""

    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "extract" in result.output.lower()

    def test_extract_help(self) -> None:
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "extract" in result.output.lower()

