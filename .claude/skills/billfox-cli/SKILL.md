---
name: billfox-cli
description: Guide for using the billfox CLI to manage receipts — adding, storing, searching, listing, editing, deleting, and configuring. Use when working with billfox commands, processing receipt/invoice documents, querying the receipt database, or troubleshooting billfox CLI usage.
---

# Billfox CLI

Receipt and invoice data extraction CLI. Processes documents through OCR and LLM parsing into a searchable SQLite database.

If `billfox` is not installed, use `uvx billfox` instead (e.g. `uvx billfox receipt list`).

## Golden Rule: Minimal Data Fetching

**ALWAYS use `--fields` to request only the data needed.** Never fetch full receipts when a subset of fields answers the question.

```bash
# User asks: "what's the vendor name of my last receipt?"
billfox receipt list --per-page 1 -f vendor_name

# User asks: "how much did I spend at Woolworths?"
billfox receipt search "Woolworths" -f total

# User asks: "show me item details from my coffee receipts"
billfox receipt search "coffee" -f vendor_name,items.description,items.total
```

**Available fields for `--fields`:**

| Category | Fields |
|---|---|
| Vendor | `vendor_name`, `vendor_business_number`, `vendor_email`, `vendor_phone`, `vendor_address`, `vendor_website` |
| Financial | `total`, `tax_amount`, `tax_rate`, `surcharge_amount`, `currency`, `payment_method` |
| Details | `expense_number`, `expense_date`, `invoice_number`, `country` |
| Collections | `items`, `tags`, `view_tags`, `expense_type` |
| Item subfields | `items.description`, `items.description_translated`, `items.total`, `items.tax_amount`, `items.tags` |

Monetary fields (`total`, `tax_amount`, `surcharge_amount`, `items.total`, `items.tax_amount`) auto-include `currency`.

## Searching with Filters

Combine text query with `--where` conditions to narrow results. Multiple `--where` flags use AND logic.

**`--where` syntax:** `"FIELD OPERATOR VALUE"`

| Fields | Operators |
|---|---|
| `total`, `tax_amount`, `surcharge_amount`, `tax_rate` | `=`, `>`, `<`, `>=`, `<=` |

```bash
# Keyword + amount filter
billfox receipt search "coffee" -w "total>50" -f vendor_name,total

# Amount range
billfox receipt search "lunch" -w "total>=20" -w "total<=100" -f vendor_name,total,expense_date

# Amount + tax filter
billfox receipt search "supplies" -w "total>100" -w "tax_amount<=10" -f vendor_name,total,tax_amount

# High tax rate
billfox receipt search "restaurant" -w "tax_rate>=0.10" -f vendor_name,total,tax_rate

# Exact amount
billfox receipt search "uber" -w "total=25.50" -f vendor_name,expense_date
```

Receipts with missing values for filtered fields are excluded. Conditions apply post-search (text query runs first, then filters).

## Sorting

Both `search` and `list` support `--sort` and `--direction` options. Default: `--sort expense_date --direction desc`.

| Sort field | Description |
|---|---|
| `expense_date` | Receipt/transaction date (default) |
| `created_at` | When receipt was added to database |
| `updated_at` | When receipt was last modified |
| `total` | Receipt total amount |

| Direction | Description |
|---|---|
| `desc` | Newest first (default) |
| `asc` | Oldest first |

```bash
# Oldest receipts first
billfox receipt list --sort expense_date --direction asc -f vendor_name,expense_date

# Recently added receipts
billfox receipt list --sort created_at -f vendor_name,expense_date

# Recently modified
billfox receipt list --sort updated_at -f vendor_name,expense_date

# Search results sorted by date instead of relevance
billfox receipt search "coffee" --sort expense_date -f vendor_name,total,expense_date
```

Receipts with null sort field values appear last regardless of direction.

## Practical Scenarios

```bash
# "What did I buy at Coles?"
billfox receipt search "Coles" -f items.description,items.total

# "How much tax did I pay on expensive purchases?"
billfox receipt search "" -w "total>200" -f vendor_name,total,tax_amount

# "Find business expenses over $100"
billfox receipt search "business" -w "total>100" -f vendor_name,total,expense_date,tags

# "What's my most recent receipt?"
billfox receipt list --per-page 1 -f vendor_name,total,expense_date

# "What's my oldest receipt?"
billfox receipt list --per-page 1 --direction asc -f vendor_name,total,expense_date

# "List all receipts from page 3"
billfox receipt list --page 3 -f vendor_name,total,expense_date

# "Find receipts with 'office' in the name, under $50"
billfox receipt search "office" -w "total<50" -f vendor_name,total

# "Show me the original file for receipt ABC123"
billfox receipt get ABC123 --original

# "Change vendor name and tag a receipt"
billfox receipt edit ABC123 --vendor-name "New Name" --tags "office,supplies"

# "Get receipt data as JSON for scripting"
billfox receipt search "vendor" --json -f vendor_name,total
```

## Search Modes

| Mode | Flag | Best for |
|---|---|---|
| Hybrid (default) | `--mode hybrid` | Balanced keyword + semantic matching |
| BM25 | `--mode bm25` | Exact keywords, invoice numbers, phrases |
| Vector | `--mode vector` | Conceptual/fuzzy queries, paraphrasing |

## Commands Reference

### `billfox receipt add <file>`

Add a receipt by parsing image/PDF through OCR + LLM and storing result.

| Flag | Description |
|---|---|
| `--model, -m` | LLM model (e.g. `openai:gpt-4.1`) |
| `--extractor, -e` | OCR: `docling` or `mistral` |
| `--preprocess, -p` | Preprocessor chain (e.g. `yolo,resize`) |
| `--store` | Database path |
| `--json, -j` | JSON output |
| `--verbose, -v` | Show progress |

### `billfox receipt list`

| Flag | Description |
|---|---|
| `--db, -d` | Database path |
| `--page, -p` | Page number (default: 1) |
| `--per-page, -n` | Results per page (default: 20) |
| `--fields, -f` | Fields to return |
| `--sort, -s` | Sort by: `created_at`, `updated_at`, `expense_date`, `total` (default: `expense_date`) |
| `--direction` | Sort direction: `asc` or `desc` (default: `desc`) |
| `--where, -w` | Numeric filter (repeatable, AND logic) |
| `--json, -j` | JSON output |

### `billfox receipt search <query>`

| Flag | Description |
|---|---|
| `--db, -d` | Database path |
| `--limit, -l` | Max results (default: 20) |
| `--mode, -m` | `hybrid`, `vector`, or `bm25` |
| `--fields, -f` | Fields to return |
| `--where, -w` | Numeric filter (repeatable, AND logic) |
| `--sort, -s` | Sort by: `created_at`, `updated_at`, `expense_date`, `total` (default: `expense_date`) |
| `--direction` | Sort direction: `asc` or `desc` (default: `desc`) |
| `--json, -j` | JSON output |

### `billfox receipt edit <document_id>`

```bash
billfox receipt edit ID --vendor-name "Name" --total 15.50
billfox receipt edit ID --data '{"tags": ["food"]}'
```

Editable flags: `--vendor-name`, `--total`, `--expense-date`, `--currency`, `--tax-amount`, `--tax-rate`, `--payment-method`, `--invoice-number`, `--tags`, `--expense-type`, `--data` (arbitrary JSON).

See Receipt Model section below for all fields.

### `billfox receipt get <document_id>`

Get file path. `--original` for pre-crop path.

### `billfox receipt delete <document_id>`

Delete receipt. Supports `--db` and `--json`.

### `billfox extract <file>`

OCR-only extraction, no LLM or storage.

```bash
billfox extract doc.pdf --extractor mistral --output result.md
```

### `billfox backup <files...>`

Backup files using configured provider.

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

### `billfox init`

Interactive setup wizard for OCR, LLM, and backup providers. Config: `~/.billfox/config.toml`.

### `billfox llms.txt`

Print LLM-consumable documentation.

## Receipt Model

**Receipt fields:** `vendor_name`, `vendor_business_number`, `vendor_email`, `vendor_phone`, `vendor_address`, `vendor_website`, `total` (float), `tax_amount` (float), `tax_rate` (float, decimal e.g. 0.10=10%), `surcharge_amount` (float), `currency` (ISO 3-letter), `payment_method`, `expense_number`, `expense_date` (ISO 8601), `invoice_number`, `country` (ISO 2-letter), `items` (list), `tags` (list), `view_tags` (list), `expense_type` ("business"/"personal").

**ReceiptItem fields:** `description`, `description_translated`, `total` (float), `tax_amount` (float), `tags` (list).
