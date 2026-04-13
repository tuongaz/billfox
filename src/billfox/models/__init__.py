"""Structured data models for billfox."""

from billfox.models.prompts import RECEIPT_SYSTEM_PROMPT
from billfox.models.receipt import Receipt, ReceiptItem

__all__ = [
    "RECEIPT_SYSTEM_PROMPT",
    "Receipt",
    "ReceiptItem",
]
