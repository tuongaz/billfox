"""SQLite-only VectorType — stores float32 vectors as raw bytes (sqlite-vec compatible)."""

from __future__ import annotations

import struct
from typing import Any

from sqlalchemy import LargeBinary
from sqlalchemy.types import TypeDecorator


class VectorType(TypeDecorator[list[float]]):
    """Vector column: stores as raw float32 bytes for sqlite-vec compatibility.

    Bind: ``list[float]`` → ``bytes`` (struct-packed float32).
    Result: ``bytes`` → ``list[float]``.
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        super().__init__()

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> bytes | None:
        if value is None:
            return None
        return struct.pack(f"{len(value)}f", *value)

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            count = len(value) // 4
            return list(struct.unpack(f"{count}f", value))
        return value  # type: ignore[no-any-return]
