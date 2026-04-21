---
name: billfox-cli
description: Guide for using the billfox CLI to manage receipts — adding, storing, searching, listing, editing, deleting, and configuring. Use when working with billfox commands, processing receipt/invoice documents, querying the receipt database, or troubleshooting billfox CLI usage.
---

# Billfox CLI

Run `billfox llms.txt` to get full CLI documentation.

If `billfox` is not installed, use `uvx billfox llms.txt` instead.

## Golden Rule: Minimal Data Fetching

**ALWAYS use `--fields` to request only the data needed.** Never fetch full receipts when a subset of fields answers the question.
