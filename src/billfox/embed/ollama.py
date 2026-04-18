from __future__ import annotations

from typing import Any

import httpx


class OllamaEmbedder:
    """Generate text embeddings using a local Ollama instance.

    Implements the Embedder protocol.

    Args:
        model: Ollama model name (e.g. ``'nomic-embed-text'``).
        base_url: Ollama server URL. Defaults to ``http://localhost:11434``.
        dimensions: Embedding vector dimensionality. When *None* the value is
            auto-detected by sending a probe request to the Ollama server.
    """

    _DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._dimensions: int | None = dimensions

    # ------------------------------------------------------------------
    # Embedder protocol
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Return embedding vector dimensionality, auto-detecting if needed."""
        if self._dimensions is None:
            self._dimensions = self._probe_dimensions()
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            One embedding vector per input text.
        """
        if not texts:
            return []

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=60.0,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        embeddings: list[list[float]] = data["embeddings"]
        return embeddings

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _probe_dimensions(self) -> int:
        """Send a short probe request to detect embedding dimensions."""
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": ["dimension probe"]},
            timeout=30.0,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        embeddings: list[list[float]] = data["embeddings"]
        if not embeddings or not embeddings[0]:
            raise RuntimeError(
                f"Ollama model {self._model!r} returned empty embeddings. "
                "Ensure the model supports embeddings."
            )
        return len(embeddings[0])
