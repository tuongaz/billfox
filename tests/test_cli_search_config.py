"""Tests for billfox CLI search and config commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from billfox._types import SearchResult
from billfox.cli.app import app

runner = CliRunner()


# ── search command ──────────────────────────────────────────────


class TestSearchCommand:
    """Tests for the search subcommand."""

    def test_search_json_output(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    document_id="doc1",
                    data={"vendor": "Acme", "total": 42.0},
                    score=0.95,
                    signals={"bm25": 0.8, "final_score": 0.95},
                ),
            ]
        )

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app, ["receipt", "search", "acme", "--db", "/tmp/test.db", "--json"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["document_id"] == "doc1"
        assert parsed[0]["score"] == 0.95

    def test_search_table_output(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    document_id="doc1",
                    data={"name": "Test"},
                    score=0.85,
                ),
            ]
        )

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app, ["receipt", "search", "test", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0
        assert "doc1" in result.output

    def test_search_no_results(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(return_value=[])

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app, ["receipt", "search", "nothing", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0

    def test_search_with_mode_bm25(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(return_value=[])

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app,
                ["receipt", "search", "query", "--db", "/tmp/test.db", "--mode", "bm25"],
            )

        assert result.exit_code == 0
        mock_store.search.assert_awaited_once_with("query", limit=20, mode="bm25")

    def test_search_with_limit(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(return_value=[])

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app,
                ["receipt", "search", "query", "--db", "/tmp/test.db", "--limit", "5"],
            )

        assert result.exit_code == 0
        mock_store.search.assert_awaited_once_with("query", limit=5, mode="hybrid")

    def test_search_multiple_results_json(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(document_id="d1", data={"a": 1}, score=0.9),
                SearchResult(document_id="d2", data={"b": 2}, score=0.7),
            ]
        )

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(
                app, ["receipt", "search", "q", "--db", "/tmp/t.db", "--json"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 2
        assert parsed[0]["score"] > parsed[1]["score"]

    def test_search_default_db_path(self) -> None:
        """Search uses ~/.billfox/receipts.db by default (--db is optional)."""
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.close = AsyncMock()
        mock_store_cls.return_value.search = AsyncMock(return_value=[])

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", mock_store_cls),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(app, ["receipt", "search", "coffee"])

        assert result.exit_code == 0
        call_kwargs = mock_store_cls.call_args[1]
        assert call_kwargs["db_path"] == str(Path.home() / ".billfox" / "receipts.db")

    def test_search_uses_receipt_model(self) -> None:
        """Search command uses Receipt model instead of generic _AnyModel."""
        from billfox.models.receipt import Receipt

        mock_store_cls = MagicMock()
        mock_store_cls.return_value.close = AsyncMock()
        mock_store_cls.return_value.search = AsyncMock(return_value=[])

        with (
            patch("billfox.store.sqlite.SQLiteDocumentStore", mock_store_cls),
            patch("billfox.cli._helpers.try_build_embedder", return_value=None),
        ):
            result = runner.invoke(app, ["receipt", "search", "test", "--db", "/tmp/t.db"])

        assert result.exit_code == 0
        call_kwargs = mock_store_cls.call_args[1]
        assert call_kwargs["schema"] is Receipt

    def test_search_help(self) -> None:
        result = runner.invoke(app, ["receipt", "search", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "--limit" in result.output
        assert "--mode" in result.output
        assert "--json" in result.output


# ── config commands ─────────────────────────────────────────────


class TestConfigCommand:
    """Tests for the config subcommands."""

    def test_config_set(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            result = runner.invoke(
                app, ["config", "set", "api_keys.mistral", "sk-test"]
            )

        assert result.exit_code == 0
        assert "Set" in result.output
        config_file = tmp_path / "config.toml"
        assert config_file.exists()

    def test_config_get(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            runner.invoke(
                app, ["config", "set", "api_keys.mistral", "sk-test"]
            )
            result = runner.invoke(
                app, ["config", "get", "api_keys.mistral"]
            )

        assert result.exit_code == 0
        assert "sk-test" in result.output

    def test_config_get_missing_key(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            result = runner.invoke(
                app, ["config", "get", "nonexistent.key"]
            )

        assert result.exit_code == 1

    def test_config_list(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            runner.invoke(
                app, ["config", "set", "api_keys.mistral", "sk-m"]
            )
            runner.invoke(
                app, ["config", "set", "api_keys.openai", "sk-o"]
            )
            result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "api_keys.mistral" in result.output
        assert "api_keys.openai" in result.output

    def test_config_list_empty(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "No configuration" in result.output

    def test_config_set_nested_creates_parents(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            runner.invoke(
                app, ["config", "set", "defaults.extractor", "mistral"]
            )
            result = runner.invoke(
                app, ["config", "get", "defaults.extractor"]
            )

        assert result.exit_code == 0
        assert "mistral" in result.output

    def test_config_set_overwrite(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            runner.invoke(
                app, ["config", "set", "api_keys.openai", "old-key"]
            )
            runner.invoke(
                app, ["config", "set", "api_keys.openai", "new-key"]
            )
            result = runner.invoke(
                app, ["config", "get", "api_keys.openai"]
            )

        assert result.exit_code == 0
        assert "new-key" in result.output

    def test_config_supports_all_keys(self, tmp_path: Path) -> None:
        with patch("billfox.cli._helpers.get_config_dir", return_value=tmp_path):
            runner.invoke(
                app, ["config", "set", "api_keys.mistral", "m-key"]
            )
            runner.invoke(
                app, ["config", "set", "api_keys.openai", "o-key"]
            )
            runner.invoke(
                app, ["config", "set", "defaults.extractor", "mistral"]
            )
            runner.invoke(
                app, ["config", "set", "defaults.model", "openai:gpt-4.1"]
            )
            result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "api_keys.mistral" in result.output
        assert "api_keys.openai" in result.output
        assert "defaults.extractor" in result.output
        assert "defaults.model" in result.output

    def test_config_dir_auto_created(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "subdir" / ".billfox"
        with patch("billfox.cli._helpers.get_config_dir", return_value=config_dir):
            result = runner.invoke(
                app, ["config", "set", "api_keys.mistral", "sk-x"]
            )

        assert result.exit_code == 0
        assert config_dir.exists()
        assert (config_dir / "config.toml").exists()


# ── help ─────────────────────────────────────────────────────────


class TestSearchConfigHelp:
    """Tests for help output of search and config commands."""

    def test_main_help_includes_receipt(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "receipt" in result.output.lower()

    def test_main_help_includes_config(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_search_help_shows_flags(self) -> None:
        result = runner.invoke(app, ["receipt", "search", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "--limit" in result.output
        assert "--mode" in result.output
        # --db should not be marked as required (has a default now)
        assert "receipts.db" in result.output

    def test_config_help(self) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "set" in result.output.lower()
        assert "get" in result.output.lower()
        assert "list" in result.output.lower()

    def test_config_set_help(self) -> None:
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code == 0

    def test_config_get_help(self) -> None:
        result = runner.invoke(app, ["config", "get", "--help"])
        assert result.exit_code == 0

    def test_config_list_help(self) -> None:
        result = runner.invoke(app, ["config", "list", "--help"])
        assert result.exit_code == 0
