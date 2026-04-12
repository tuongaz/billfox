"""Tests for billfox CLI extract and parse commands."""

from __future__ import annotations

import json
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
            result = runner.invoke(app, ["extract", str(img)])

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
            result = runner.invoke(app, ["extract", str(img), "--output", str(out)])

        assert result.exit_code == 0
        assert out.read_text() == "Extracted markdown"

    def test_extract_missing_file(self) -> None:
        mock_source = MagicMock()
        mock_source.load = AsyncMock(side_effect=FileNotFoundError("File not found: /nonexistent"))

        with patch("billfox.source.local.LocalFileSource", return_value=mock_source):
            result = runner.invoke(app, ["extract", "/nonexistent.jpg"])

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
            result = runner.invoke(app, ["extract", str(img), "--preprocess", "resize"])

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
            result = runner.invoke(app, ["extract", str(img), "--verbose"])

        assert result.exit_code == 0


# ── parse command ────────────────────────────────────────────────


class TestParseCommand:
    """Tests for the parse subcommand."""

    def test_parse_outputs_json_to_stdout(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Invoice(BaseModel):\n"
            "    vendor_name: str\n"
            "    total: float\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"vendor_name": "Acme", "total": 42.0}
        mock_result.model_dump_json.return_value = '{"vendor_name": "Acme", "total": 42.0}'

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch("billfox.pipeline.Pipeline", return_value=mock_pipeline):
            result = runner.invoke(
                app,
                ["parse", str(img), "--schema", f"{schema_file}:Invoice"],
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "Acme"
        assert parsed["total"] == 42.0

    def test_parse_with_json_flag(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Invoice(BaseModel):\n"
            "    vendor_name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"vendor_name": "Acme"}
        mock_result.model_dump_json.return_value = '{"vendor_name":"Acme"}'

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch("billfox.pipeline.Pipeline", return_value=mock_pipeline):
            result = runner.invoke(
                app,
                ["parse", str(img), "--schema", f"{schema_file}:Invoice", "--json"],
            )

        assert result.exit_code == 0
        assert "Acme" in result.output

    def test_parse_writes_to_output_file(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        out = tmp_path / "output.json"

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}
        mock_result.model_dump_json.return_value = '{"name": "Widget"}'

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch("billfox.pipeline.Pipeline", return_value=mock_pipeline):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--output", str(out),
                ],
            )

        assert result.exit_code == 0
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["name"] == "Widget"

    def test_parse_invalid_schema_format(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        result = runner.invoke(
            app,
            ["parse", str(img), "--schema", "no_colon_here"],
        )
        assert result.exit_code != 0

    def test_parse_schema_file_not_found(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        result = runner.invoke(
            app,
            ["parse", str(img), "--schema", "/nonexistent/schema.py:Foo"],
        )
        assert result.exit_code != 0

    def test_parse_schema_class_not_found(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Foo(BaseModel):\n"
            "    x: int\n"
        )

        result = runner.invoke(
            app,
            ["parse", str(img), "--schema", f"{schema_file}:Bar"],
        )
        assert result.exit_code != 0

    def test_parse_with_store(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        db_path = tmp_path / "test.db"

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}
        mock_result.model_dump_json.return_value = '{"name": "Widget"}'

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch("billfox.pipeline.Pipeline", return_value=mock_pipeline) as mock_cls:
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--store", str(db_path),
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("store") is not None

    def test_parse_with_custom_model_and_prompt(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "X"}
        mock_result.model_dump_json.return_value = '{"name": "X"}'

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.parse.llm.LLMParser") as mock_parser_cls,
        ):
            mock_parser_cls.return_value = MagicMock()
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--model", "anthropic:claude-sonnet-4-20250514",
                    "--prompt", "Extract items from this receipt.",
                ],
            )

        assert result.exit_code == 0
        mock_parser_cls.assert_called_once()
        call_kwargs = mock_parser_cls.call_args
        assert call_kwargs.kwargs["model"] == "anthropic:claude-sonnet-4-20250514"
        assert call_kwargs.kwargs["system_prompt"] == "Extract items from this receipt."


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

    def test_parse_help(self) -> None:
        result = runner.invoke(app, ["parse", "--help"])
        assert result.exit_code == 0
        assert "schema" in result.output.lower()


# ── schema loader ────────────────────────────────────────────────


class TestLoadSchema:
    """Tests for _load_schema helper."""

    def test_load_valid_schema(self, tmp_path: Path) -> None:
        from billfox.cli.app import _load_schema

        schema_file = tmp_path / "myschema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Receipt(BaseModel):\n"
            "    total: float\n"
        )

        cls = _load_schema(f"{schema_file}:Receipt")
        assert cls.__name__ == "Receipt"

    def test_load_schema_no_colon(self) -> None:
        import typer

        from billfox.cli.app import _load_schema

        try:
            _load_schema("noformat")
            msg = "Should have raised"
            raise AssertionError(msg)
        except typer.BadParameter:
            pass

    def test_load_schema_missing_file(self) -> None:
        import typer

        from billfox.cli.app import _load_schema

        try:
            _load_schema("/nonexistent.py:Foo")
            msg = "Should have raised"
            raise AssertionError(msg)
        except typer.BadParameter:
            pass

    def test_load_schema_missing_class(self, tmp_path: Path) -> None:
        import typer

        from billfox.cli.app import _load_schema

        schema_file = tmp_path / "schema.py"
        schema_file.write_text("x = 1\n")

        try:
            _load_schema(f"{schema_file}:Missing")
            msg = "Should have raised"
            raise AssertionError(msg)
        except typer.BadParameter:
            pass
