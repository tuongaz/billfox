from __future__ import annotations

import asyncio
import base64
import os
import struct
from typing import Any

# Known dimensions for OpenAI embedding models
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Generate text embeddings using the OpenAI API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "No OpenAI API key provided. Pass api_key= or set OPENAI_API_KEY."
            )
        self._model = model

    @property
    def dimensions(self) -> int:
        return _MODEL_DIMENSIONS.get(self._model, 1536)

    def _get_client(self) -> Any:
        try:
            import openai  # noqa: I001
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedder. "
                "Install it with: pip install billfox[openai]"
            ) from None
        return openai.AsyncOpenAI(api_key=self._api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()

        async def _get_one(text: str) -> list[float]:
            response = await client.embeddings.create(
                model=self._model,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding  # type: ignore[no-any-return]

        results: list[list[float]] = list(
            await asyncio.gather(*[_get_one(t) for t in texts])
        )
        return results


def encode_vector(vector: list[float]) -> str:
    """Pack a float vector into a base64-encoded string."""
    return base64.b64encode(struct.pack(f"{len(vector)}f", *vector)).decode("ascii")


def decode_vector(encoded: str) -> list[float]:
    """Decode a base64-encoded string back into a float vector."""
    raw = base64.b64decode(encoded)
    count = len(raw) // 4
    return list(struct.unpack(f"{count}f", raw))
