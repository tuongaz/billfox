"""CLI command to output LLM-consumable documentation of all billfox commands."""

from __future__ import annotations

import sys

import typer


def llms_txt() -> None:
    """Print Markdown documentation of all billfox commands for LLM consumption."""
    sys.stdout.write(_LLMS_TXT)


_LLMS_TXT = """\
# billfox

Composable document data extraction: load, preprocess, OCR, LLM parse, store with vector search.

## Commands

### `billfox init`

Interactive setup wizard. Configures OCR provider, LLM provider, and backup provider. Writes configuration to `~/.billfox/config.toml`.

Options:
- `--yes, -y` — Skip confirmation when overwriting existing config.

### `billfox extract <file>`

Extract markdown from a document using OCR.

Arguments:
- `file` (required) — Path to the document file to extract.

Options:
- `--extractor, -e` — Extractor to use: `docling` (default) or `mistral`.
- `--preprocess, -p` — Comma-separated preprocessors (e.g. `resize`).
- `--api-key` — API key for the extractor.
- `--output, -o` — Output file path. Defaults to stdout.
- `--verbose, -v` — Enable debug output.

### `billfox backup <files...>`

Back up files using the configured backup provider (local folder or Google Drive).

Arguments:
- `files` (required) — One or more file paths to back up.

### `billfox config set <key> <value>`

Set a configuration value using dot-notation keys.

Arguments:
- `key` (required) — Config key (e.g. `api_keys.mistral`).
- `value` (required) — Config value.

### `billfox config get <key>`

Get a configuration value.

Arguments:
- `key` (required) — Config key (e.g. `api_keys.mistral`).

### `billfox config list`

List all configuration values.

### `billfox auth google-drive`

Authorize billfox to access your Google Drive. Opens a browser for OAuth flow.

Options:
- `--force, -f` — Re-authorize even if already authorized.

### `billfox auth status`

Show which integrations are currently authorized.

### `billfox receipt add <file>`

Add a receipt by parsing it into structured data using OCR + LLM. Automatically crops receipts with YOLO object detection and resizes before extraction when no `--preprocess` is specified.

Arguments:
- `file` (required) — Path to the receipt file to parse.

Options:
- `--model, -m` — LLM model identifier (reads from config if not set).
- `--extractor, -e` — Extractor to use: `docling` (default) or `mistral`.
- `--preprocess, -p` — Comma-separated preprocessors (e.g. `resize`).
- `--api-key` — API key for the extractor.
- `--store` — SQLite database path (defaults to `~/.billfox/receipts.db`).
- `--output, -o` — Output file path. Defaults to stdout.
- `--json, -j` — Output machine-readable JSON.
- `--verbose, -v` — Enable debug output.

### `billfox receipt search <query>`

Search stored receipts using hybrid, vector, or BM25 search.

Arguments:
- `query` (required) — Search query.

Options:
- `--db, -d` — SQLite database path (defaults to `~/.billfox/receipts.db`).
- `--limit, -l` — Maximum number of results (default: 20).
- `--mode, -m` — Search mode: `hybrid` (default), `vector`, or `bm25`.
- `--fields, -f` — Comma-separated fields to include in output. Supports top-level fields (e.g. `vendor_name,total`) and nested item fields with dot notation (e.g. `items.description,items.total`). Use `items` for full item objects. When monetary fields (total, tax_amount, surcharge_amount) or item monetary fields (items.total, items.tax_amount) are requested, currency is auto-included.
- `--where, -w` — Filter condition. Repeatable (AND logic). Operators: `=`, `>`, `<`, `>=`, `<=`. Supported fields: `total`, `tax_amount`, `surcharge_amount`, `tax_rate`, `expense_date`. Numeric examples: `--where 'total>50'`, `--where 'total>=100' --where 'tax_amount<=10'`. Date examples: `--where 'expense_date>=2024-01-01'`, `--where 'expense_date>=2024-06-01' --where 'expense_date<2024-07-01'`.
- `--sort, -s` — Sort by: `expense_date` (default), `created_at`, `updated_at`, or `total`.
- `--direction` — Sort direction: `desc` (default) or `asc`.
- `--json, -j` — Output machine-readable JSON.
- `--verbose, -v` — Enable debug output.

### `billfox receipt list`

List stored receipts with pagination.

Options:
- `--db, -d` — SQLite database path (defaults to `~/.billfox/receipts.db`).
- `--page, -p` — Page number, starts at 1 (default: 1).
- `--per-page, -n` — Items per page (default: 20).
- `--fields, -f` — Comma-separated fields to include in output. Supports top-level fields (e.g. `vendor_name,total`) and nested item fields with dot notation (e.g. `items.description,items.total`). Use `items` for full item objects. When monetary fields (total, tax_amount, surcharge_amount) or item monetary fields (items.total, items.tax_amount) are requested, currency is auto-included.
- `--where, -w` — Filter condition. Repeatable (AND logic). Operators: `=`, `>`, `<`, `>=`, `<=`. Supported fields: `total`, `tax_amount`, `surcharge_amount`, `tax_rate`, `expense_date`. Numeric examples: `--where 'total>50'`. Date examples: `--where 'expense_date>=2024-01-01'`.
- `--sort, -s` — Sort by: `expense_date` (default), `created_at`, `updated_at`, or `total`.
- `--direction` — Sort direction: `desc` (default) or `asc`.
- `--json, -j` — Output machine-readable JSON.
- `--verbose, -v` — Enable debug output.

### `billfox receipt get <document_id>`

Print the file path of a stored receipt's cropped or original file.

Arguments:
- `document_id` (required) — Receipt document ID.

Options:
- `--original, -o` — Get the original (pre-crop) file instead.
- `--db, -d` — SQLite database path (defaults to `~/.billfox/receipts.db`).

### `billfox receipt delete <document_id>`

Delete a stored receipt by document ID.

Arguments:
- `document_id` (required) — Receipt document ID to delete.

Options:
- `--db, -d` — SQLite database path (defaults to `~/.billfox/receipts.db`).
- `--json, -j` — Output machine-readable JSON.
- `--verbose, -v` — Enable debug output.

### `billfox receipt edit <document_id>`

Edit fields of a stored receipt. Pass updates via `--data` JSON or individual field flags.

Arguments:
- `document_id` (required) — Receipt document ID to edit.

Options:
- `--data` — JSON string with fields to update.
- `--vendor-name` — Vendor name.
- `--total` — Total amount.
- `--expense-date` — Expense date.
- `--currency` — Currency code.
- `--tax-amount` — Tax amount.
- `--tax-rate` — Tax rate.
- `--payment-method` — Payment method.
- `--invoice-number` — Invoice number.
- `--tags` — Comma-separated tags.
- `--db, -d` — SQLite database path (defaults to `~/.billfox/receipts.db`).
- `--json, -j` — Output machine-readable JSON.
- `--verbose, -v` — Enable debug output.

### `billfox llms.txt`

Print this documentation in Markdown format for LLM consumption.

## Configuration

billfox stores configuration in `~/.billfox/config.toml`. Run `billfox init` to set up interactively.

Environment variables are loaded from `~/.billfox/.env` (global) and `./.env` (project-local). Existing env vars take precedence.

### Supported providers

**OCR:** `docling` (local, free) or `mistral` (API, requires `MISTRAL_API_KEY`).

**LLM:** `openai` (requires `OPENAI_API_KEY`), `anthropic` (requires `ANTHROPIC_API_KEY`), or `ollama` (local, no key needed).

**Backup:** `local` (saves to a configured folder) or `google_drive` (requires OAuth via `billfox auth google-drive`).
"""
