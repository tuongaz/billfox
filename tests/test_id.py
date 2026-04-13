"""Tests for billfox._id module."""

from __future__ import annotations

import re

from billfox._id import generate_id


def test_generate_id_returns_string() -> None:
    assert isinstance(generate_id(), str)


def test_generate_id_is_hex() -> None:
    result = generate_id()
    assert re.fullmatch(r"[0-9a-f]{32}", result), f"Expected 32-char hex, got {result!r}"


def test_generate_id_unique() -> None:
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100
