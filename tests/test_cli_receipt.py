"""Tests for billfox CLI receipt command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from billfox.cli.app import app

runner = CliRunner()


def _make_test_file(tmp_path: Path) -> Path:
    """Create a minimal test document file."""
    f = tmp_path / "test.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0fake")
    return f


def _mock_receipt_result() -> MagicMock:
    """Create a mock Receipt result."""
    mock_result = MagicMock()
    mock_result.vendor_name = "Coffee Shop"
    mock_result.total = 5.50
    mock_result.items = []
    mock_result.model_dump.return_value = {
        "vendor_name": "Coffee Shop",
        "total": 5.50,
        "currency": "AUD",
        "items": [],
        "tags": ["food & drink"],
        "view_tags": [],
    }
    mock_result.model_dump_json.return_value = json.dumps(
        mock_result.model_dump.return_value, indent=2,
    )
    return mock_result


class TestReceiptCommand:
    """Tests for the receipt subcommand."""

    def test_receipt_outputs_json_to_stdout(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "Coffee Shop"
        assert parsed["total"] == 5.50

    def test_receipt_with_json_flag(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1", "--json"])

        assert result.exit_code == 0
        assert "Coffee Shop" in result.output

    def test_receipt_writes_to_output_file(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)
        out = tmp_path / "output.json"

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, [
                "receipt", "parse", str(img),
                "--model", "openai:gpt-4.1",
                "--output", str(out),
            ])

        assert result.exit_code == 0
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["vendor_name"] == "Coffee Shop"

    def test_receipt_uses_receipt_model_and_prompt(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_llm_parser_cls = MagicMock()

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser", mock_llm_parser_cls),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 0
        call_kwargs = mock_llm_parser_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4.1"

        from billfox.models.prompts import RECEIPT_SYSTEM_PROMPT
        from billfox.models.receipt import Receipt

        assert call_kwargs["system_prompt"] == RECEIPT_SYSTEM_PROMPT
        assert call_kwargs["output_type"] is Receipt

    def test_receipt_defaults_store_to_receipts_db(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_store_cls = MagicMock()
        mock_store_cls.return_value.close = AsyncMock()
        mock_store_cls.return_value.delete = AsyncMock()

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore", mock_store_cls),
        ):
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 0
        call_kwargs = mock_store_cls.call_args.kwargs
        assert call_kwargs["db_path"].endswith("receipts.db")

    def test_receipt_explicit_store_path(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)
        db_path = str(tmp_path / "custom.db")

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_store_cls = MagicMock()
        mock_store_cls.return_value.close = AsyncMock()
        mock_store_cls.return_value.delete = AsyncMock()

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore", mock_store_cls),
        ):
            result = runner.invoke(app, [
                "receipt", "parse", str(img),
                "--model", "openai:gpt-4.1",
                "--store", db_path,
            ])

        assert result.exit_code == 0
        call_kwargs = mock_store_cls.call_args.kwargs
        assert call_kwargs["db_path"] == db_path

    def test_receipt_uses_configured_ocr_provider(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "mistral"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
            patch("billfox.cli._helpers.build_extractor") as mock_build_ext,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            mock_build_ext.return_value = MagicMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 0
        mock_build_ext.assert_called_once_with("mistral", None)

    def test_receipt_model_override(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_llm_parser_cls = MagicMock()

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {
                    "ocr": {"provider": "docling"},
                    "llm": {"provider": "openai", "model": "openai:gpt-4.1"},
                },
            }),
            patch("billfox.parse.llm.LLMParser", mock_llm_parser_cls),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, [
                "receipt", "parse", str(img),
                "--model", "anthropic:claude-sonnet-4-20250514",
            ])

        assert result.exit_code == 0
        call_kwargs = mock_llm_parser_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic:claude-sonnet-4-20250514"

    def test_receipt_error_handling(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=FileNotFoundError("File not found"))

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_receipt_requires_config(self) -> None:
        with patch("billfox.cli._helpers.read_config", return_value={}):
            result = runner.invoke(app, ["receipt", "parse", "/some/file.jpg"])

        assert result.exit_code == 1
        assert "not configured" in result.output.lower()

    def test_receipt_uses_ulid_as_id(self, tmp_path: Path) -> None:
        img = _make_test_file(tmp_path)

        mock_result = _mock_receipt_result()
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline),
            patch("billfox.cli._helpers.read_config", return_value={
                "defaults": {"ocr": {"provider": "docling"}},
            }),
            patch("billfox.parse.llm.LLMParser"),
            patch("billfox.store.sqlite.SQLiteDocumentStore") as mock_store_cls,
            patch("billfox._id.generate_id", return_value="01abc123def456"),
        ):
            mock_store_cls.return_value.close = AsyncMock()
            mock_store_cls.return_value.delete = AsyncMock()
            result = runner.invoke(app, ["receipt", "parse", str(img), "--model", "openai:gpt-4.1"])

        assert result.exit_code == 0
        mock_pipeline.run.assert_awaited_once()
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs["document_id"] == "01abc123def456"


class TestReceiptListCommand:
    """Tests for the receipt list subcommand."""

    def test_list_json_output(self) -> None:
        mock_data = MagicMock()
        mock_data.model_dump.return_value = {
            "vendor_name": "Acme", "total": 42.0, "currency": "AUD",
        }

        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.list_documents = AsyncMock(
            return_value=([("doc1", mock_data)], 1),
        )

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "list", "--db", "/tmp/test.db", "--json"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["total"] == 1
        assert parsed["page"] == 1
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["document_id"] == "doc1"

    def test_list_empty(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.list_documents = AsyncMock(return_value=([], 0))

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "list", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0

    def test_list_pagination_params(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.list_documents = AsyncMock(return_value=([], 0))

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "list", "--db", "/tmp/t.db", "--page", "3", "--per-page", "10"]
            )

        assert result.exit_code == 0
        mock_store.list_documents.assert_awaited_once_with(limit=10, offset=20)

    def test_list_json_pagination_metadata(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.list_documents = AsyncMock(return_value=([], 25))

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "list", "--db", "/tmp/t.db", "--json", "--per-page", "10"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["total"] == 25
        assert parsed["total_pages"] == 3
        assert parsed["per_page"] == 10

    def test_list_help(self) -> None:
        result = runner.invoke(app, ["receipt", "list", "--help"])
        assert result.exit_code == 0
        assert "--page" in result.output
        assert "--per-page" in result.output
        assert "--json" in result.output
        assert "--db" in result.output


class TestReceiptGetCommand:
    """Tests for the receipt get subcommand."""

    def test_get_returns_file_path(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get_file_paths = AsyncMock(
            return_value=("/backups/2025/06/15/receipt.jpg", "/backups/2025/06/15/receipt_original.jpg"),
        )

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "get", "doc1", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0
        assert result.output.strip() == "/backups/2025/06/15/receipt.jpg"

    def test_get_returns_original_path(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get_file_paths = AsyncMock(
            return_value=("/backups/2025/06/15/receipt.jpg", "/backups/2025/06/15/receipt_original.jpg"),
        )

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "get", "doc1", "--original", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0
        assert result.output.strip() == "/backups/2025/06/15/receipt_original.jpg"

    def test_get_no_path_stored(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get_file_paths = AsyncMock(return_value=(None, None))

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "get", "doc1", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 1

    def test_get_help(self) -> None:
        result = runner.invoke(app, ["receipt", "get", "--help"])
        assert result.exit_code == 0
        assert "--original" in result.output
        assert "--db" in result.output


class TestReceiptDeleteCommand:
    """Tests for the receipt delete subcommand."""

    def test_delete_success(self) -> None:
        mock_receipt = _mock_receipt_result()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_receipt)
        mock_store.delete = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "delete", "doc1", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0
        assert "Deleted" in result.output
        mock_store.delete.assert_awaited_once_with("doc1")

    def test_delete_not_found(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=None)
        mock_store.delete = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "delete", "nonexistent", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
        mock_store.delete.assert_not_awaited()

    def test_delete_json_output(self) -> None:
        mock_receipt = _mock_receipt_result()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_receipt)
        mock_store.delete = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "delete", "doc1", "--db", "/tmp/test.db", "--json"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["deleted"] is True
        assert parsed["document_id"] == "doc1"

    def test_delete_not_found_json(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=None)

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "delete", "nonexistent", "--db", "/tmp/test.db", "--json"]
            )

        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["error"] == "not_found"

    def test_delete_help(self) -> None:
        result = runner.invoke(app, ["receipt", "delete", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "--json" in result.output


class TestReceiptEditCommand:
    """Tests for the receipt edit subcommand."""

    def _make_existing_receipt(self) -> MagicMock:
        from billfox.models.receipt import Receipt

        receipt = Receipt(
            vendor_name="Coffee Shop",
            total=5.50,
            currency="AUD",
            tags=["food & drink"],
        )
        return receipt

    def test_edit_single_field(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, ["receipt", "edit", "doc1", "--vendor-name", "Tea House", "--db", "/tmp/test.db"]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "Tea House"
        assert parsed["total"] == 5.50  # unchanged
        mock_store.save.assert_awaited_once()

    def test_edit_multiple_fields(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "doc1",
                    "--vendor-name", "Tea House",
                    "--total", "12.00",
                    "--currency", "USD",
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "Tea House"
        assert parsed["total"] == 12.0
        assert parsed["currency"] == "USD"

    def test_edit_with_json_data(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "doc1",
                    "--data", '{"vendor_name": "Tea House", "total": 9.99}',
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "Tea House"
        assert parsed["total"] == 9.99

    def test_edit_flags_override_json_data(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "doc1",
                    "--data", '{"vendor_name": "From JSON", "total": 1.0}',
                    "--vendor-name", "From Flag",
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["vendor_name"] == "From Flag"  # flag wins
        assert parsed["total"] == 1.0  # from --data

    def test_edit_tags(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "doc1",
                    "--tags", "travel, business",
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["tags"] == ["travel", "business"]

    def test_edit_not_found(self) -> None:
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=None)

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "nonexistent",
                    "--vendor-name", "X",
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_edit_no_updates(self) -> None:
        result = runner.invoke(
            app, ["receipt", "edit", "doc1", "--db", "/tmp/test.db"]
        )

        assert result.exit_code == 1
        assert "no updates" in result.output.lower()

    def test_edit_invalid_json_data(self) -> None:
        result = runner.invoke(
            app, ["receipt", "edit", "doc1", "--data", "not-json", "--db", "/tmp/test.db"]
        )

        assert result.exit_code == 1
        assert "invalid json" in result.output.lower()

    def test_edit_help(self) -> None:
        result = runner.invoke(app, ["receipt", "edit", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--vendor-name" in result.output
        assert "--total" in result.output
        assert "--tags" in result.output
        assert "--db" in result.output
        assert "--json" in result.output
        assert "--expense-type" in result.output

    def test_edit_expense_type(self) -> None:
        existing = self._make_existing_receipt()
        mock_store = MagicMock()
        mock_store.close = AsyncMock()
        mock_store.get = AsyncMock(return_value=existing)
        mock_store.save = AsyncMock()

        with patch("billfox.store.sqlite.SQLiteDocumentStore", return_value=mock_store):
            result = runner.invoke(
                app, [
                    "receipt", "edit", "doc1",
                    "--expense-type", "business",
                    "--db", "/tmp/test.db",
                ]
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["expense_type"] == "business"

    def test_edit_expense_type_invalid(self) -> None:
        result = runner.invoke(
            app, [
                "receipt", "edit", "doc1",
                "--expense-type", "invalid",
                "--db", "/tmp/test.db",
            ]
        )
        assert result.exit_code == 1
        assert "business" in result.output.lower() or "personal" in result.output.lower()


class TestReceiptHelp:
    """Tests for receipt help output."""

    def test_receipt_parse_help(self) -> None:
        result = runner.invoke(app, ["receipt", "parse", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--output" in result.output
        assert "--json" in result.output
        assert "--store" in result.output
        assert "--extractor" in result.output
        assert "--preprocess" in result.output
        assert "--api-key" in result.output
        assert "--verbose" in result.output

    def test_receipt_subapp_help(self) -> None:
        result = runner.invoke(app, ["receipt", "--help"])
        assert result.exit_code == 0
        assert "parse" in result.output
        assert "search" in result.output
        assert "list" in result.output
        assert "get" in result.output
        assert "delete" in result.output
        assert "edit" in result.output
