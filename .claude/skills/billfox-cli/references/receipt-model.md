# Receipt Data Model

## Receipt

| Field | Type | Description |
|---|---|---|
| `expense_number` | `str \| None` | Internal expense ID |
| `expense_date` | `str \| None` | ISO 8601 date (YYYY-MM-DD) |
| `invoice_number` | `str \| None` | Receipt/invoice number |
| `vendor_name` | `str \| None` | Business/trading name |
| `vendor_business_number` | `str \| None` | ABN, VAT, EIN, etc. |
| `vendor_email` | `str \| None` | Vendor email |
| `vendor_phone` | `str \| None` | Vendor phone |
| `vendor_address` | `str \| None` | Vendor address |
| `vendor_website` | `str \| None` | Vendor website |
| `total` | `float \| None` | Final amount paid (inclusive of tax) |
| `tax_amount` | `float \| None` | Tax amount |
| `tax_rate` | `float \| None` | Decimal (0.10 = 10%) |
| `surcharge_amount` | `float \| None` | Service fee, card surcharge |
| `payment_method` | `str \| None` | Cash, Visa, AMEX, etc. |
| `currency` | `str \| None` | 3-letter ISO code (AUD, USD, EUR) |
| `country` | `str \| None` | 2-letter ISO code (AU, US) |
| `items` | `list[ReceiptItem]` | List of purchased items |
| `tags` | `list[str]` | Expense categories (e.g. "food & drink") |
| `view_tags` | `list[str]` | UI display tags (e.g. "tax-deductible") |

Source: `src/billfox/models/receipt.py`

## ReceiptItem

| Field | Type | Description |
|---|---|---|
| `description` | `str \| None` | Item name |
| `description_translated` | `str \| None` | English translation if non-English |
| `total` | `float \| None` | Line item total |
| `tax_amount` | `float \| None` | Per-item tax |
| `tags` | `list[str]` | Item-specific tags |

## Editable Fields via CLI

When using `billfox receipt edit`, these fields can be set with flags:

- `--vendor-name` / `--total` / `--expense-date` / `--currency`
- `--tax-amount` / `--tax-rate` / `--payment-method` / `--invoice-number`
- `--tags` (comma-separated)
- `--data` (arbitrary JSON for any field)
