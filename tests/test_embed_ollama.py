from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from billfox.embed import Embedder
from billfox.embed.ollama import OllamaEmbedder

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    assert embedder._model == "nomic-embed-text"
    assert embedder._base_url == "http://localhost:11434"
    assert embedder._dimensions is None


def test_init_custom_base_url() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://remote:11434")
    assert embedder._base_url == "http://remote:11434"


def test_init_strips_trailing_slash() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://remote:11434/")
    assert embedder._base_url == "http://remote:11434"


def test_init_explicit_dimensions() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)
    assert embedder._dimensions == 768
    assert embedder.dimensions == 768


# ---------------------------------------------------------------------------
# dimensions (auto-detection via probe)
# ---------------------------------------------------------------------------


def test_dimensions_probes_when_not_set() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    probe_vector = [0.1] * 768

    mock_response = httpx.Response(
        200,
        json={"embeddings": [probe_vector]},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch("httpx.post", return_value=mock_response) as mock_post:
        dims = embedder.dimensions

    assert dims == 768
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["dimension probe"]},
        timeout=30.0,
    )


def test_dimensions_cached_after_probe() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    probe_vector = [0.1] * 768

    mock_response = httpx.Response(
        200,
        json={"embeddings": [probe_vector]},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch("httpx.post", return_value=mock_response) as mock_post:
        _ = embedder.dimensions
        _ = embedder.dimensions  # second access should not call probe again

    assert mock_post.call_count == 1


def test_dimensions_raises_on_empty_embeddings() -> None:
    embedder = OllamaEmbedder(model="bad-model")

    mock_response = httpx.Response(
        200,
        json={"embeddings": [[]]},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="empty embeddings"):
            _ = embedder.dimensions


def test_dimensions_skips_probe_when_explicit() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=384)
    with patch("httpx.post") as mock_post:
        dims = embedder.dimensions
    assert dims == 384
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_empty_list() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_single_text() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    expected = [[0.1, 0.2, 0.3]]

    mock_response = httpx.Response(
        200,
        json={"embeddings": expected},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        result = await embedder.embed(["hello"])

    assert result == expected
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["hello"]},
        timeout=60.0,
    )


@pytest.mark.asyncio
async def test_embed_multiple_texts() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    mock_response = httpx.Response(
        200,
        json={"embeddings": expected},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        result = await embedder.embed(["a", "b", "c"])

    assert result == expected


@pytest.mark.asyncio
async def test_embed_uses_custom_base_url() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://remote:11434")

    mock_response = httpx.Response(
        200,
        json={"embeddings": [[0.1]]},
        request=httpx.Request("POST", "http://remote:11434/api/embed"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        await embedder.embed(["test"])

    mock_post.assert_called_once_with(
        "http://remote:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["test"]},
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)
    assert isinstance(embedder, Embedder)
