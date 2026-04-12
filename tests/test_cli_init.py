"""Tests for billfox init wizard and config guard (US-010)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from billfox._types import ExtractionResult
from billfox.cli.app import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Init wizard: Docling + OpenAI + Local backup
# ---------------------------------------------------------------------------


class TestInitDoclingOpenAILocal:
    """Selecting Docling + OpenAI + Local backup writes correct config."""

    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_writes_correct_config_with_all_nested_keys(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        backup_path = str(tmp_path / "backups")
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input=f"1\n1\n1\n{backup_path}\n",
        )

        assert result.exit_code == 0
        mock_write.assert_called_once()
        config = mock_write.call_args[0][0]

        # OCR
        assert config["defaults"]["ocr"]["provider"] == "docling"
        # LLM
        assert config["defaults"]["llm"]["provider"] == "openai"
        assert config["defaults"]["llm"]["model"] == "openai:gpt-4.1"
        # Backup
        assert config["defaults"]["backup"]["provider"] == "local"
        assert config["defaults"]["backup"]["local_path"] == backup_path
        # No ollama section
        assert "ollama" not in config["defaults"]

    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_shows_openai_env_var_guidance(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        backup_path = str(tmp_path / "backups")
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input=f"1\n1\n1\n{backup_path}\n",
        )

        assert result.exit_code == 0
        assert "OPENAI_API_KEY" in result.output


# ---------------------------------------------------------------------------
# Init wizard: Mistral + Claude + Google Drive
# ---------------------------------------------------------------------------


class TestInitMistralClaudeGDrive:
    """Selecting Mistral + Claude + Google Drive writes correct config."""

    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_writes_correct_config(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="2\n2\n2\n",
        )

        assert result.exit_code == 0
        mock_write.assert_called_once()
        config = mock_write.call_args[0][0]

        assert config["defaults"]["ocr"]["provider"] == "mistral"
        assert config["defaults"]["llm"]["provider"] == "anthropic"
        assert config["defaults"]["llm"]["model"] == "anthropic:claude-sonnet-4-20250514"
        assert config["defaults"]["backup"]["provider"] == "google_drive"
        assert "local_path" not in config["defaults"]["backup"]
        assert "ollama" not in config["defaults"]

    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_shows_mistral_and_anthropic_env_guidance(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="2\n2\n2\n",
        )

        assert result.exit_code == 0
        assert "MISTRAL_API_KEY" in result.output
        assert "ANTHROPIC_API_KEY" in result.output

    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_shows_google_drive_auth_instruction(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="2\n2\n2\n",
        )

        assert result.exit_code == 0
        assert "billfox auth google-drive" in result.output


# ---------------------------------------------------------------------------
# Init wizard: Ollama selection
# ---------------------------------------------------------------------------


class TestInitOllama:
    """Selecting Ollama triggers base URL prompt and stores ollama config."""

    @patch("billfox.cli.init._check_ollama", return_value=None)
    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_stores_base_url_and_model(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        backup_path = str(tmp_path / "backups")
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input=f"1\n3\nhttp://myhost:11434\nmymodel\n1\n{backup_path}\n",
        )

        assert result.exit_code == 0
        config = mock_write.call_args[0][0]

        assert config["defaults"]["llm"]["provider"] == "ollama"
        assert config["defaults"]["llm"]["model"] == "ollama:mymodel"
        assert config["defaults"]["ollama"]["base_url"] == "http://myhost:11434"
        assert config["defaults"]["ollama"]["model"] == "mymodel"

    @patch("billfox.cli.init._check_ollama", return_value=None)
    @patch("billfox.cli.app._write_config")
    @patch("billfox.cli.app._read_config", return_value={})
    def test_no_api_key_guidance_for_ollama(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        backup_path = str(tmp_path / "backups")
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input=f"1\n3\nhttp://localhost:11434\nllama3.2\n1\n{backup_path}\n",
        )

        assert result.exit_code == 0
        assert "No API keys required" in result.output


# ---------------------------------------------------------------------------
# Init wizard: Overwrite confirmation
# ---------------------------------------------------------------------------


class TestInitOverwriteConfirmation:
    """Re-running init with existing config asks for confirmation."""

    @patch("billfox.cli.app._write_config")
    @patch(
        "billfox.cli.app._read_config",
        return_value={"defaults": {"ocr": {"provider": "docling"}}},
    )
    def test_decline_overwrite_cancels_setup(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        result = runner.invoke(app, ["init"], input="n\n")

        assert result.exit_code == 0
        assert "Existing configuration found" in result.output
        mock_write.assert_not_called()

    @patch("billfox.cli.app._write_config")
    @patch(
        "billfox.cli.app._read_config",
        return_value={"defaults": {"ocr": {"provider": "docling"}}},
    )
    def test_confirm_overwrite_proceeds_with_wizard(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        # y=overwrite, 1=Docling, 1=OpenAI, 2=Google Drive
        result = runner.invoke(app, ["init"], input="y\n1\n1\n2\n")

        assert result.exit_code == 0
        mock_write.assert_called_once()

    @patch("billfox.cli.app._write_config")
    @patch(
        "billfox.cli.app._read_config",
        return_value={"defaults": {"ocr": {"provider": "docling"}}},
    )
    def test_yes_flag_skips_confirmation(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        # --yes skips overwrite prompt; 1=Docling, 1=OpenAI, 2=Google Drive
        result = runner.invoke(app, ["init", "--yes"], input="1\n1\n2\n")

        assert result.exit_code == 0
        mock_write.assert_called_once()
        assert "Existing configuration found" not in result.output


# ---------------------------------------------------------------------------
# Config guard: triggers when config is missing
# ---------------------------------------------------------------------------


class TestConfigGuardTriggers:
    """Config guard triggers on extract/parse/backup when config is missing."""

    @patch("billfox.cli.app._read_config", return_value={})
    def test_extract_guard_triggers(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        result = runner.invoke(app, ["extract", str(img)])

        assert result.exit_code == 1
        assert "billfox is not configured" in result.output

    @patch("billfox.cli.app._read_config", return_value={})
    def test_parse_guard_triggers(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        result = runner.invoke(
            app,
            ["parse", str(img), "--schema", f"{schema_file}:Item"],
        )

        assert result.exit_code == 1
        assert "billfox is not configured" in result.output

    @patch("billfox.cli.app._read_config", return_value={})
    def test_backup_guard_triggers(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        result = runner.invoke(app, ["backup", str(f)])

        assert result.exit_code == 1
        assert "billfox is not configured" in result.output


# ---------------------------------------------------------------------------
# Config guard: bypassed with explicit CLI flags
# ---------------------------------------------------------------------------


class TestConfigGuardBypass:
    """Config guard does NOT trigger with explicit --extractor or --model flags."""

    @patch("billfox.cli.app._read_config", return_value={})
    def test_extract_bypassed_with_explicit_extractor(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_source = MagicMock()
        mock_source.load = AsyncMock(
            return_value=MagicMock(content=b"bytes", mime_type="image/jpeg"),
        )
        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractionResult(markdown="text", pages=[]),
        )

        with (
            patch("billfox.source.local.LocalFileSource", return_value=mock_source),
            patch(
                "billfox.extract.docling.DoclingExtractor",
                return_value=mock_extractor,
            ),
        ):
            result = runner.invoke(
                app,
                ["extract", str(img), "--extractor", "docling"],
            )

        assert result.exit_code == 0
        assert "billfox is not configured" not in result.output

    @patch("billfox.cli.app._read_config", return_value={})
    def test_parse_bypassed_with_explicit_model(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.extract.docling.DoclingExtractor"),
        ):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--model", "openai:gpt-4.1",
                ],
            )

        assert result.exit_code == 0
        assert "billfox is not configured" not in result.output

    @patch("billfox.cli.app._read_config", return_value={})
    def test_parse_bypassed_with_explicit_extractor(
        self,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.extract.docling.DoclingExtractor"),
        ):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--extractor", "docling",
                ],
            )

        assert result.exit_code == 0
        assert "billfox is not configured" not in result.output
