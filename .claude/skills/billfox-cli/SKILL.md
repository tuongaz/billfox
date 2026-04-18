---
name: billfox-cli
description: Guide for using the billfox CLI to manage receipts — parsing, storing, searching, listing, editing, deleting, and configuring. Use when working with billfox commands, processing receipt/invoice documents, querying the receipt database, or troubleshooting billfox CLI usage.
---

# Billfox CLI

Receipt and invoice data extraction CLI. Processes documents through OCR and LLM parsing into a searchable SQLite database.

## Quick Start

```bash
billfox init              # configure OCR, LLM, and backup providers
billfox receipt parse invoice.jpg   # parse and store a receipt
billfox receipt list                # list stored receipts
billfox receipt search "coffee"     # search receipts
```

Config: `~/.billfox/config.toml`. API keys: `~/.billfox/.env` or `./.env`.

## Commands

### `billfox receipt parse <file>`

Parse a receipt image/PDF and store the extracted data.

```bash
billfox receipt parse receipt.jpg
billfox receipt parse receipt.jpg --model openai:gpt-4.1 --json
billfox receipt parse receipt.jpg --extractor mistral --verbose
```

| Flag | Description |
|---|---|
| `--model` | LLM model (e.g. `openai:gpt-4.1`) |
| `--extractor` | OCR provider: `docling` or `mistral` |
| `--preprocess` | Preprocessor chain (e.g. `yolo,resize`) |
| `--store` | Database path (default: `~/.billfox/receipts.db`) |
| `--json` | JSON output |
| `--verbose` | Show progress |

### `billfox receipt list`

```bash
billfox receipt list
billfox receipt list --page 2 --per-page 10
billfox receipt list -f vendor_name,total,date --json
```

| Flag | Description |
|---|---|
| `--db` | Database path |
| `--page` | Page number (default: 1) |
| `--per-page` | Results per page (default: 20) |
| `--fields, -f` | Fields to return (comma-separated, dot notation for items e.g. `items.description`) |
| `--json` | JSON output |

### `billfox receipt search <query>`

```bash
billfox receipt search "coffee"
billfox receipt search "lunch" --mode bm25 --limit 5
billfox receipt search "coffee" --where "total>50"
billfox receipt search "lunch" -w "total>=20" -w "total<=100"
billfox receipt search "supplies" -w "total>100" -w "tax_amount<=10"
```

| Flag | Description |
|---|---|
| `--db` | Database path |
| `--limit` | Max results (default: 20) |
| `--mode` | `hybrid` (default), `vector`, or `bm25` |
| `--fields, -f` | Fields to return (comma-separated) |
| `--where, -w` | Numeric filter (repeatable, AND logic) |
| `--json` | JSON output |

**`--where` syntax:** `"FIELD OPERATOR VALUE"` where fields are `total`, `tax_amount`, `surcharge_amount`, `tax_rate` and operators are `=`, `>`, `<`, `>=`, `<=`.

### `billfox receipt get <document_id>`

Get receipt file path. Use `--original` for pre-crop path.

### `billfox receipt edit <document_id>`

```bash
billfox receipt edit abc123 --vendor-name "Coffee Shop" --total 15.50
billfox receipt edit abc123 --data '{"tags": ["food & drink"]}'
```

Editable fields: see [references/receipt-model.md](references/receipt-model.md).

### `billfox receipt delete <document_id>`

Delete a receipt. Supports `--db` and `--json`.

### `billfox extract <file>`

OCR-only extraction (no LLM parsing or storage). Useful for debugging.

```bash
billfox extract doc.pdf --extractor mistral --output result.md
```

### `billfox backup <files...>`

Backup files using configured provider (local or Google Drive).

### `billfox auth`

```bash
billfox auth google-drive [--force]
billfox auth status
```

### `billfox config`

```bash
billfox config set defaults.llm.model openai:gpt-4.1
billfox config get defaults.ocr.provider
billfox config list
```

Keys use dot notation: `defaults.llm.model`, `defaults.ocr.provider`, `api_keys.openai`.

### `billfox llms.txt`

Print LLM-consumable documentation about billfox.
