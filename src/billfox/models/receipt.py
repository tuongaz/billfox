"""Receipt and ReceiptItem models for structured receipt data extraction."""

from __future__ import annotations

from pydantic import BaseModel


class ReceiptItem(BaseModel):
    """A single line item on a receipt."""

    description: str | None = None
    description_translated: str | None = None
    total: float | None = None
    tax_amount: float | None = None
    tags: list[str] = []


class Receipt(BaseModel):
    """Structured receipt data extracted from a document."""

    expense_number: str | None = None
    expense_date: str | None = None
    vendor_name: str | None = None
    vendor_business_number: str | None = None
    vendor_email: str | None = None
    vendor_phone: str | None = None
    vendor_address: str | None = None
    vendor_website: str | None = None
    total: float | None = None
    tax_amount: float | None = None
    tax_rate: float | None = None
    surcharge_amount: float | None = None
    invoice_number: str | None = None
    payment_method: str | None = None
    currency: str | None = None
    country: str | None = None
    items: list[ReceiptItem] = []
    tags: list[str] = []
    view_tags: list[str] = []
    expense_type: str = "personal"

    def search_text(self) -> str:
        """Build composite text for search indexing (BM25 + vector)."""
        parts: list[str] = []
        if self.vendor_name:
            parts.append(f"Vendor: {self.vendor_name}")
        if self.invoice_number:
            parts.append(f"Invoice: {self.invoice_number}")
        if self.items:
            descs = ", ".join(
                item.description for item in self.items if item.description
            )
            if descs:
                parts.append(f"Items: {descs}")
        if self.tags:
            parts.append(f"Tags: {' '.join(self.tags)}")
        if self.expense_type:
            parts.append(f"Expense type: {self.expense_type}")
        return "\n".join(parts)
