"""Tests for CLI parse command LLM config wiring (US-006)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from billfox.cli.app import app

runner = CliRunner()


def _make_schema_file(tmp_path: Path) -> Path:
    """Create a minimal Pydantic schema file for testing."""
    schema_file = tmp_path / "schema.py"
    schema_file.write_text(
        "from pydantic import BaseModel\n\n"
        "class Item(BaseModel):\n"
        "    name: str\n"
    )
    return schema_file


def _make_test_file(tmp_path: Path) -> Path:
    """Create a minimal test document file."""
    f = tmp_path / "test.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0fake")
    return f


def _run_parse(
    tmp_path: Path,
    config: dict,
    extra_args: list[str] | None = None,
) -> tuple:
    """Run parse command with mocked config and pipeline, return (result, Pipeline call kwargs)."""
    img = _make_test_file(tmp_path)
    schema_file = _make_schema_file(tmp_path)

    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"name": "Widget"}

    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=mock_result)

    mock_llm_parser_cls = MagicMock()

    args = [
        "parse", str(img),
        "--schema", f"{schema_file}:Item",
    ]
    if extra_args:
        args.extend(extra_args)

    with (
        patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
        patch("billfox.cli._helpers.read_config", return_value=config),
        patch("billfox.parse.llm.LLMParser", mock_llm_parser_cls) as mock_cls,
    ):
        result = runner.invoke(app, args)

    return result, mock_cls


class TestParseOllamaConfigWiring:
    """Tests that parse command reads Ollama config correctly."""

    def test_ollama_provider_constructs_model_string(self, tmp_path: Path) -> None:
        config = {
            "defaults": {
                "ocr": {"provider": "docling"},
                "llm": {"provider": "ollama"},
                "ollama": {"model": "llama3.2:7b", "base_url": "http://myhost:11434"},
            },
        }
        result, mock_cls = _run_parse(tmp_path, config)

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "ollama:llama3.2:7b"
        assert call_kwargs["base_url"] == "http://myhost:11434"

    def test_ollama_provider_default_base_url_is_none_when_not_in_config(
        self, tmp_path: Path,
    ) -> None:
        config = {
            "defaults": {
                "ocr": {"provider": "docling"},
                "llm": {"provider": "ollama"},
                "ollama": {"model": "mistral"},
            },
        }
        result, mock_cls = _run_parse(tmp_path, config)

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "ollama:mistral"
        # base_url should be None (LLMParser handles its own default)
        assert call_kwargs["base_url"] is None

    def test_cli_model_override_takes_precedence(self, tmp_path: Path) -> None:
        config = {
            "defaults": {
                "llm": {"provider": "ollama"},
                "ollama": {"model": "llama3.2:7b"},
            },
        }
        result, mock_cls = _run_parse(
            tmp_path, config, extra_args=["--model", "openai:gpt-4.1"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4.1"

    def test_cli_ollama_override_picks_up_base_url_from_config(
        self, tmp_path: Path,
    ) -> None:
        config = {
            "defaults": {
                "llm": {"provider": "openai", "model": "openai:gpt-4.1"},
                "ollama": {"base_url": "http://remote:11434"},
            },
        }
        result, mock_cls = _run_parse(
            tmp_path, config, extra_args=["--model", "ollama:phi3"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "ollama:phi3"
        assert call_kwargs["base_url"] == "http://remote:11434"


class TestParseNonOllamaConfigWiring:
    """Tests that parse command reads non-Ollama LLM config correctly."""

    def test_config_llm_model_used_as_default(self, tmp_path: Path) -> None:
        config = {
            "defaults": {
                "ocr": {"provider": "docling"},
                "llm": {"provider": "openai", "model": "openai:gpt-4.1-mini"},
            },
        }
        result, mock_cls = _run_parse(tmp_path, config)

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4.1-mini"
        assert call_kwargs["base_url"] is None

    def test_anthropic_model_from_config(self, tmp_path: Path) -> None:
        config = {
            "defaults": {
                "ocr": {"provider": "docling"},
                "llm": {
                    "provider": "anthropic",
                    "model": "anthropic:claude-sonnet-4-20250514",
                },
            },
        }
        result, mock_cls = _run_parse(tmp_path, config)

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic:claude-sonnet-4-20250514"

    def test_fallback_to_default_when_no_config(self, tmp_path: Path) -> None:
        config = {"defaults": {"ocr": {"provider": "docling"}}}
        result, mock_cls = _run_parse(tmp_path, config)

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4.1"

    def test_cli_model_overrides_config_model(self, tmp_path: Path) -> None:
        config = {
            "defaults": {
                "ocr": {"provider": "docling"},
                "llm": {"provider": "openai", "model": "openai:gpt-4.1-mini"},
            },
        }
        result, mock_cls = _run_parse(
            tmp_path, config, extra_args=["--model", "anthropic:claude-sonnet-4-20250514"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic:claude-sonnet-4-20250514"
