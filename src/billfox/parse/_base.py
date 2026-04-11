from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Parser(Protocol[T_co]):
    """Protocol for parsing markdown into structured data."""

    async def parse(self, markdown: str) -> T_co: ...
