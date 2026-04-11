from __future__ import annotations

import pytest

from billfox._types import Document
from billfox.preprocess.chain import PreprocessorChain


class _AppendMetaPreprocessor:
    """Test preprocessor that appends a key to metadata."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value

    async def process(self, document: Document) -> Document:
        return Document(
            content=document.content,
            mime_type=document.mime_type,
            source_uri=document.source_uri,
            metadata={**document.metadata, self._key: self._value},
        )


class _UpperCaseContentPreprocessor:
    """Test preprocessor that uppercases content bytes (ASCII)."""

    async def process(self, document: Document) -> Document:
        return Document(
            content=document.content.upper(),
            mime_type=document.mime_type,
            source_uri=document.source_uri,
            metadata=document.metadata,
        )


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        content=b"hello",
        mime_type="text/plain",
        source_uri="test.txt",
    )


@pytest.mark.asyncio
async def test_chain_applies_in_order(sample_doc: Document) -> None:
    """Preprocessors are applied sequentially in the given order."""
    chain = PreprocessorChain([
        _AppendMetaPreprocessor("step", "1"),
        _AppendMetaPreprocessor("step", "2"),
    ])

    result = await chain.process(sample_doc)

    # The second preprocessor overwrites the first's "step" key
    assert result.metadata["step"] == "2"


@pytest.mark.asyncio
async def test_chain_order_matters(sample_doc: Document) -> None:
    """Different ordering produces different results."""
    chain_a = PreprocessorChain([
        _AppendMetaPreprocessor("first", "a"),
        _AppendMetaPreprocessor("second", "b"),
    ])
    chain_b = PreprocessorChain([
        _AppendMetaPreprocessor("second", "b"),
        _AppendMetaPreprocessor("first", "a"),
    ])

    result_a = await chain_a.process(sample_doc)
    result_b = await chain_b.process(sample_doc)

    # Both have the same keys, but insertion order differs
    assert result_a.metadata == {"first": "a", "second": "b"}
    assert result_b.metadata == {"second": "b", "first": "a"}


@pytest.mark.asyncio
async def test_chain_empty_returns_original(sample_doc: Document) -> None:
    """An empty chain returns the original document unchanged."""
    chain = PreprocessorChain([])

    result = await chain.process(sample_doc)

    assert result is sample_doc


@pytest.mark.asyncio
async def test_chain_single_preprocessor(sample_doc: Document) -> None:
    """A chain with one preprocessor works like using it directly."""
    chain = PreprocessorChain([_UpperCaseContentPreprocessor()])

    result = await chain.process(sample_doc)

    assert result.content == b"HELLO"


@pytest.mark.asyncio
async def test_chain_composes_content_and_metadata(sample_doc: Document) -> None:
    """Chain composes both content and metadata transforms."""
    chain = PreprocessorChain([
        _UpperCaseContentPreprocessor(),
        _AppendMetaPreprocessor("processed", "true"),
    ])

    result = await chain.process(sample_doc)

    assert result.content == b"HELLO"
    assert result.metadata["processed"] == "true"


@pytest.mark.asyncio
async def test_chain_implements_preprocessor_protocol() -> None:
    """PreprocessorChain satisfies the Preprocessor protocol."""
    from billfox.preprocess import Preprocessor

    chain = PreprocessorChain([])
    assert isinstance(chain, Preprocessor)
