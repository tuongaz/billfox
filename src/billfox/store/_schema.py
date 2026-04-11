"""SQLAlchemy ORM models for document storage."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    """Stores serialised Pydantic model data for a document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_name: Mapped[str] = mapped_column(String, nullable=False)
    data_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    embeddings: Mapped[list[DocumentEmbeddingRow]] = relationship(
        "DocumentEmbeddingRow",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentEmbeddingRow(Base):
    """Per-field embedding storage for a document."""

    __tablename__ = "document_embeddings"
    __table_args__ = (
        UniqueConstraint("document_id", "field_name", name="uq_doc_field"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    field_name: Mapped[str] = mapped_column(String, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)

    document: Mapped[DocumentRow] = relationship(
        "DocumentRow",
        back_populates="embeddings",
    )
