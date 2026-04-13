"""Tests for Receipt and ReceiptItem Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from billfox.models.receipt import Receipt, ReceiptItem


class TestReceiptItem:
    def test_defaults(self) -> None:
        item = ReceiptItem()
        assert item.description is None
        assert item.description_translated is None
        assert item.total is None
        assert item.tax_amount is None
        assert item.tags == []

    def test_all_fields(self) -> None:
        item = ReceiptItem(
            description="Coffee",
            description_translated="Café",
            total=4.50,
            tax_amount=0.45,
            tags=["food", "beverage"],
        )
        assert item.description == "Coffee"
        assert item.description_translated == "Café"
        assert item.total == 4.50
        assert item.tax_amount == 0.45
        assert item.tags == ["food", "beverage"]

    def test_serialization_roundtrip(self) -> None:
        item = ReceiptItem(description="Milk", total=3.00, tags=["grocery"])
        data = item.model_dump()
        restored = ReceiptItem.model_validate(data)
        assert restored == item

    def test_json_roundtrip(self) -> None:
        item = ReceiptItem(description="Bread", total=2.50)
        json_str = item.model_dump_json()
        restored = ReceiptItem.model_validate_json(json_str)
        assert restored == item


class TestReceipt:
    def test_minimal(self) -> None:
        receipt = Receipt(is_expense=True)
        assert receipt.is_expense is True
        assert receipt.items == []
        assert receipt.tags == []
        assert receipt.view_tags == []
        assert receipt.vendor_name is None
        assert receipt.total is None

    def test_is_expense_required(self) -> None:
        with pytest.raises(ValidationError):
            Receipt()  # type: ignore[call-arg]

    def test_all_fields(self) -> None:
        receipt = Receipt(
            is_expense=True,
            invalid_receipt_reason=None,
            expense_number="EXP-001",
            expense_date="2026-01-15",
            vendor_name="Acme Corp",
            vendor_business_number="ABN 12345678901",
            vendor_email="info@acme.com",
            vendor_phone="+61 2 1234 5678",
            vendor_address="123 Main St, Sydney",
            vendor_website="https://acme.com",
            total=110.00,
            tax_amount=10.00,
            tax_rate=10.0,
            surcharge_amount=1.50,
            invoice_number="INV-2026-001",
            payment_method="credit_card",
            currency="AUD",
            country="Australia",
            items=[
                ReceiptItem(description="Widget", total=50.00),
                ReceiptItem(description="Gadget", total=50.00),
            ],
            tags=["office", "supplies"],
            view_tags=["Q1-2026"],
        )
        assert receipt.vendor_name == "Acme Corp"
        assert receipt.total == 110.00
        assert len(receipt.items) == 2
        assert receipt.items[0].description == "Widget"
        assert receipt.tags == ["office", "supplies"]
        assert receipt.view_tags == ["Q1-2026"]
        assert receipt.currency == "AUD"
        assert receipt.country == "Australia"
        assert receipt.surcharge_amount == 1.50

    def test_not_expense(self) -> None:
        receipt = Receipt(
            is_expense=False,
            invalid_receipt_reason="Not a receipt - this is a menu",
        )
        assert receipt.is_expense is False
        assert receipt.invalid_receipt_reason == "Not a receipt - this is a menu"

    def test_serialization_roundtrip(self) -> None:
        receipt = Receipt(
            is_expense=True,
            vendor_name="Store",
            total=25.00,
            items=[ReceiptItem(description="Item A", total=25.00)],
            tags=["retail"],
        )
        data = receipt.model_dump()
        restored = Receipt.model_validate(data)
        assert restored == receipt

    def test_json_roundtrip(self) -> None:
        receipt = Receipt(
            is_expense=True,
            vendor_name="Cafe",
            total=15.50,
            currency="USD",
            items=[ReceiptItem(description="Latte", total=5.50, tags=["coffee"])],
        )
        json_str = receipt.model_dump_json()
        restored = Receipt.model_validate_json(json_str)
        assert restored == receipt

    def test_defaults_for_optional_fields(self) -> None:
        receipt = Receipt(is_expense=True)
        assert receipt.invalid_receipt_reason is None
        assert receipt.expense_number is None
        assert receipt.expense_date is None
        assert receipt.vendor_name is None
        assert receipt.vendor_business_number is None
        assert receipt.vendor_email is None
        assert receipt.vendor_phone is None
        assert receipt.vendor_address is None
        assert receipt.vendor_website is None
        assert receipt.total is None
        assert receipt.tax_amount is None
        assert receipt.tax_rate is None
        assert receipt.surcharge_amount is None
        assert receipt.invoice_number is None
        assert receipt.payment_method is None
        assert receipt.currency is None
        assert receipt.country is None


class TestImportability:
    def test_import_from_models_package(self) -> None:
        from billfox.models import Receipt, ReceiptItem

        assert Receipt is not None
        assert ReceiptItem is not None

    def test_import_from_receipt_module(self) -> None:
        from billfox.models.receipt import Receipt, ReceiptItem

        assert Receipt is not None
        assert ReceiptItem is not None
