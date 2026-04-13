"""Unique document ID generation using ULID."""

from __future__ import annotations

from ulid import ULID


def generate_id() -> str:
    """Generate a unique, time-sortable document ID as a ULID hex string."""
    return ULID().hex
