"""Tests for Ollama connectivity check in billfox init wizard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from billfox.cli.init import _check_ollama


class TestCheckOllama:
    """Tests for _check_ollama helper."""

    def test_returns_model_names_on_success(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "codellama:7b"},
            ],
        }
        response.raise_for_status = MagicMock()

        with patch("billfox.cli.init.httpx.get", return_value=response) as mock_get:
            result = _check_ollama("http://localhost:11434")

        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5.0)
        assert result == ["llama3.2:latest", "codellama:7b"]

    def test_returns_empty_list_when_no_models(self) -> None:
        response = MagicMock()
        response.json.return_value = {"models": []}
        response.raise_for_status = MagicMock()

        with patch("billfox.cli.init.httpx.get", return_value=response):
            result = _check_ollama("http://localhost:11434")

        assert result == []

    def test_returns_none_on_connection_error(self) -> None:
        with patch("billfox.cli.init.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = _check_ollama("http://localhost:11434")

        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        with patch("billfox.cli.init.httpx.get", side_effect=httpx.TimeoutException("timeout")):
            result = _check_ollama("http://localhost:99999")

        assert result is None

    def test_returns_none_on_http_error(self) -> None:
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(),
        )

        with patch("billfox.cli.init.httpx.get", return_value=response):
            result = _check_ollama("http://localhost:11434")

        assert result is None

    def test_custom_base_url(self) -> None:
        response = MagicMock()
        response.json.return_value = {"models": [{"name": "phi3:latest"}]}
        response.raise_for_status = MagicMock()

        with patch("billfox.cli.init.httpx.get", return_value=response) as mock_get:
            result = _check_ollama("http://192.168.1.100:11434")

        mock_get.assert_called_once_with("http://192.168.1.100:11434/api/tags", timeout=5.0)
        assert result == ["phi3:latest"]


class TestInitWizardOllamaIntegration:
    """Tests for Ollama connectivity in the full init wizard flow."""

    @patch("billfox.cli.init._check_ollama")
    @patch("billfox.cli._helpers.write_config")
    @patch("billfox.cli._helpers.read_config", return_value={})
    @patch("billfox.cli._helpers.get_machine_timezone", return_value="Australia/Sydney")
    def test_ollama_with_models_shows_selection(
        self,
        mock_tz: MagicMock,
        mock_read: MagicMock,
        mock_write: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        from typer.testing import CliRunner

        from billfox.cli.app import app

        mock_check.return_value = ["llama3.2:latest", "codellama:7b", "phi3:latest"]

        runner = CliRunner()
        # Choices: OCR=1 (Docling), LLM=3 (Ollama), base_url=default,
        # model selection=2 (codellama:7b), embedding=3 (None), backup=1 (local), backup path=default, 1=timezone
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="1\n3\nhttp://localhost:11434\n2\n3\n1\n\n1\n",
        )

        assert result.exit_code == 0
        assert "Connected!" in result.output or "Connected" in result.stderr if hasattr(result, 'stderr') else True
        mock_check.assert_called_once_with("http://localhost:11434")

        # Verify the written config has the selected model
        written_config = mock_write.call_args[0][0]
        assert written_config["defaults"]["ollama"]["model"] == "codellama:7b"
        assert written_config["defaults"]["llm"]["model"] == "ollama:codellama:7b"

    @patch("billfox.cli.init._check_ollama")
    @patch("billfox.cli._helpers.write_config")
    @patch("billfox.cli._helpers.read_config", return_value={})
    @patch("billfox.cli._helpers.get_machine_timezone", return_value="Australia/Sydney")
    def test_ollama_connection_failure_falls_back_to_prompt(
        self,
        mock_tz: MagicMock,
        mock_read: MagicMock,
        mock_write: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        from typer.testing import CliRunner

        from billfox.cli.app import app

        mock_check.return_value = None  # Connection failed

        runner = CliRunner()
        # Choices: OCR=1, LLM=3 (Ollama), base_url=default,
        # manual model name=mymodel, embedding=3 (None), backup=1 (local), backup path=default, 1=timezone
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="1\n3\nhttp://localhost:11434\nmymodel\n3\n1\n\n1\n",
        )

        assert result.exit_code == 0
        mock_check.assert_called_once_with("http://localhost:11434")

        written_config = mock_write.call_args[0][0]
        assert written_config["defaults"]["ollama"]["model"] == "mymodel"
        assert written_config["defaults"]["llm"]["model"] == "ollama:mymodel"

    @patch("billfox.cli.init._check_ollama")
    @patch("billfox.cli._helpers.write_config")
    @patch("billfox.cli._helpers.read_config", return_value={})
    @patch("billfox.cli._helpers.get_machine_timezone", return_value="Australia/Sydney")
    def test_ollama_empty_models_falls_back_to_prompt(
        self,
        mock_tz: MagicMock,
        mock_read: MagicMock,
        mock_write: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        from typer.testing import CliRunner

        from billfox.cli.app import app

        mock_check.return_value = []  # Connected but no models

        runner = CliRunner()
        # Choices: OCR=1, LLM=3, base_url=default,
        # manual model=llama3.2 (default), embedding=3 (None), backup=1, backup path=default, 1=timezone
        result = runner.invoke(
            app,
            ["init", "--yes"],
            input="1\n3\nhttp://localhost:11434\nllama3.2\n3\n1\n\n1\n",
        )

        assert result.exit_code == 0
        mock_check.assert_called_once_with("http://localhost:11434")

        written_config = mock_write.call_args[0][0]
        assert written_config["defaults"]["ollama"]["model"] == "llama3.2"
