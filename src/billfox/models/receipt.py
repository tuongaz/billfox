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

    is_expense: bool
    invalid_receipt_reason: str | None = None
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
