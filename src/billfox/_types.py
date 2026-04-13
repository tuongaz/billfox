from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """A single page of extracted content."""

    index: int
    markdown: str


@dataclass(frozen=True)
class Document:
    """A loaded document with raw content and metadata."""

    content: bytes
    mime_type: str
    source_uri: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Result of document extraction (OCR/conversion to markdown)."""

    markdown: str
    pages: list[Page]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A single search result with scoring signals."""

    document_id: str
    data: dict[str, object]
    score: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BackupResult:
    """Result of backing up a document to a remote location."""

    uri: str
    provider: str
    original_uri: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
