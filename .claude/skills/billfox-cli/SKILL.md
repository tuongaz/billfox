---
name: billfox-cli
description: Guide for using the billfox CLI to manage receipts — parsing, storing, searching, listing, editing, deleting, and configuring. Use when working with billfox commands, processing receipt/invoice documents, querying the receipt database, or troubleshooting billfox CLI usage.
---

# Billfox CLI

Composable document data extraction CLI for processing receipts and invoices. Built with Typer, Pydantic, and a pipeline architecture (load -> preprocess -> OCR -> LLM parse -> store).

## Setup

```bash
billfox init
```

Interactive wizard that configures:
- **OCR provider**: `docling` (local, free) or `mistral` (API)
- **LLM provider**: `openai`, `anthropic`, or `ollama`
- **Backup provider**: `local` or `google_drive`

Config stored at `~/.billfox/config.toml`. API keys loaded from `~/.billfox/.env` or `./.env`.

## Receipt Commands

### Parse a receipt

```bash
billfox receipt parse <file> [options]
```

Runs the full pipeline: preprocess (YOLO crop + resize) -> OCR -> LLM extraction -> store to SQLite.

| Flag | Description |
|---|---|
| `--model` | LLM model override (e.g. `openai:gpt-4.1`) |
| `--extractor` | OCR provider: `docling` or `mistral` |
| `--preprocess` | Preprocessor chain (e.g. `yolo,resize`) — default: auto YOLO + resize |
| `--store` | SQLite database path (default: `~/.billfox/receipts.db`) |
| `--json` | Output as JSON |
| `--verbose` | Show detailed progress |

### List receipts

```bash
billfox receipt list [options]
```

| Flag | Description |
|---|---|
| `--db` | Database path |
| `--page` | Page number (default: 1) |
| `--per-page` | Results per page (default: 20) |
| `--fields, -f` | Comma-separated fields to return. Supports dot notation for item subfields (e.g. `items.description`). Use `items` for full item objects. Monetary fields auto-include currency. |
| `--json` | Output as JSON |

### Search receipts

```bash
billfox receipt search <query> [options]
```

| Flag | Description |
|---|---|
| `--db` | Database path |
| `--limit` | Max results (default: 20) |
| `--mode` | Search mode: `hybrid` (default), `vector`, `bm25` |
| `--fields, -f` | Comma-separated fields to return. Supports dot notation for item subfields (e.g. `items.description`). Use `items` for full item objects. Monetary fields auto-include currency. |
| `--where, -w` | Filter by numeric condition. Repeatable (AND logic). Operators: `=`, `>`, `<`, `>=`, `<=`. Fields: `total`, `tax_amount`, `surcharge_amount`, `tax_rate`. |
| `--json` | Output as JSON |

#### Searching by condition with `--where`

Use `--where` (or `-w`) to filter search results by numeric conditions. This lets you narrow results beyond just the text query — e.g. find receipts matching "coffee" that also cost over $50.

**Syntax:** `--where "FIELD OPERATOR VALUE"`

| Supported fields | Operators |
|---|---|
| `total`, `tax_amount`, `surcharge_amount`, `tax_rate` | `=`, `>`, `<`, `>=`, `<=` |

**Examples:**

```bash
# Receipts over $50
billfox receipt search "coffee" --where "total>50"

# Receipts between $20 and $100
billfox receipt search "lunch" --where "total>=20" --where "total<=100"

# Exact amount
billfox receipt search "uber" --where "total=25.50"

# Combine amount + tax conditions
billfox receipt search "supplies" --where "total>100" --where "tax_amount<=10"

# High tax rate receipts
billfox receipt search "restaurant" --where "tax_rate>=10"
```

**Behavior:**
- Multiple `--where` flags combine with AND logic (all conditions must be true)
- Receipts with `None`/missing values for a filtered field are automatically excluded
- Conditions are applied post-search — the text query runs first, then results are filtered
- Invalid field names or malformed conditions raise an error

Search modes:
- `hybrid` — BM25 + vector with Reciprocal Rank Fusion
- `bm25` — Keyword text search (FTS5)
- `vector` — Semantic similarity (requires OpenAI embeddings)

### Get receipt file path

```bash
billfox receipt get <document_id> [--original] [--db]
```

`--original` returns the pre-crop file path instead of the processed one.

### Edit a receipt

```bash
billfox receipt edit <document_id> [options]
```

Edit with individual flags or `--data` for arbitrary JSON:

```bash
billfox receipt edit abc123 --vendor-name "Coffee Shop" --total 15.50
billfox receipt edit abc123 --data '{"tags": ["food & drink"]}'
```

For the full list of editable fields, see [references/receipt-model.md](references/receipt-model.md).

### Delete a receipt

```bash
billfox receipt delete <document_id> [--db] [--json]
```

## Other Commands

### Extract markdown from a document

```bash
billfox extract <file> [--extractor docling|mistral] [--output file.md]
```

OCR extraction only — no LLM parsing or storage. Useful for debugging extraction quality.

### Backup files

```bash
billfox backup <files...>
```

Uses the configured backup provider (local directory or Google Drive).

### Google Drive auth

```bash
billfox auth google-drive [--force]
billfox auth status
```

### Configuration

```bash
billfox config set <key> <value>
billfox config get <key>
billfox config list
```

Keys use dot notation: `defaults.llm.model`, `defaults.ocr.provider`, `api_keys.openai`.

### LLM documentation

```bash
billfox llms.txt
```

Prints LLM-consumable documentation about billfox.

## Key Architecture

- **Pipeline**: `src/billfox/pipeline.py` — orchestrates load -> preprocess -> extract -> parse -> store
- **CLI**: `src/billfox/cli/` — Typer app with sub-apps for receipt, config, auth, backup
- **Models**: `src/billfox/models/receipt.py` — Pydantic Receipt/ReceiptItem models
- **Store**: `src/billfox/store/sqlite.py` — SQLite with FTS5 + sqlite-vec
- **LLM Parser**: `src/billfox/parse/llm.py` — pydantic-ai based structured extraction
- **Prompts**: `src/billfox/models/prompts.py` — LLM system prompt for receipt parsing

## Database

Default location: `~/.billfox/receipts.db` (SQLite). Override with `--store` or `--db`.

Document IDs are ULIDs. Each receipt stores:
- Serialized JSON data
- File path (processed) and original file path
- FTS5 text index for BM25 search
- Optional vector embeddings for semantic search

## Receipt Data Model

For the full receipt field reference, see [references/receipt-model.md](references/receipt-model.md).
