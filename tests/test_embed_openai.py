from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from billfox.embed import Embedder
from billfox.embed.openai import OpenAIEmbedder, decode_vector, encode_vector

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_init_with_explicit_api_key() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    assert embedder._api_key == "sk-test"
    assert embedder._model == "text-embedding-3-small"


def test_init_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    embedder = OpenAIEmbedder()
    assert embedder._api_key == "sk-env"


def test_init_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No OpenAI API key"):
        OpenAIEmbedder()


def test_init_custom_model() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
    assert embedder._model == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# dimensions property
# ---------------------------------------------------------------------------


def test_dimensions_small() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
    assert embedder.dimensions == 1536


def test_dimensions_large() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
    assert embedder.dimensions == 3072


def test_dimensions_unknown_model() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test", model="custom-model")
    assert embedder.dimensions == 1536  # default fallback


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_empty_list() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_single_text() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    expected = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=expected)]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(embedder, "_get_client", return_value=mock_client):
        result = await embedder.embed(["hello"])

    assert result == [expected]
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="hello",
        encoding_format="float",
    )


@pytest.mark.asyncio
async def test_embed_multiple_texts() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    call_count = 0

    async def mock_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        resp = MagicMock()
        resp.data = [MagicMock(embedding=vectors[call_count])]
        call_count += 1
        return resp

    mock_client = MagicMock()
    mock_client.embeddings.create = mock_create

    with patch.object(embedder, "_get_client", return_value=mock_client):
        result = await embedder.embed(["a", "b", "c"])

    assert result == vectors


# ---------------------------------------------------------------------------
# encode_vector / decode_vector
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip() -> None:
    vec = [1.0, 2.5, -3.14, 0.0, 42.0]
    encoded = encode_vector(vec)
    decoded = decode_vector(encoded)
    assert len(decoded) == len(vec)
    for a, b in zip(vec, decoded, strict=True):
        assert abs(a - b) < 1e-6


def test_encode_vector_format() -> None:
    vec = [1.0, 2.0]
    encoded = encode_vector(vec)
    raw = struct.pack("2f", 1.0, 2.0)
    import base64

    assert encoded == base64.b64encode(raw).decode("ascii")


def test_decode_empty_vector() -> None:
    encoded = encode_vector([])
    decoded = decode_vector(encoded)
    assert decoded == []


# ---------------------------------------------------------------------------
# Import error
# ---------------------------------------------------------------------------


def test_import_error_message() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    with patch.dict("sys.modules", {"openai": None}), pytest.raises(ImportError, match="openai is required"):
        embedder._get_client()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    embedder = OpenAIEmbedder(api_key="sk-test")
    assert isinstance(embedder, Embedder)
