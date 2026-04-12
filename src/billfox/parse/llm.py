from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMParser(Generic[T]):
    """Parse markdown into structured data using an LLM.

    Implements the Parser[T] protocol.

    Args:
        model: LLM model identifier (e.g. ``'openai:gpt-4.1'``,
            ``'ollama:llama3.2:7b'``).
        output_type: A Pydantic BaseModel subclass to parse into.
        system_prompt: System prompt guiding the LLM extraction.
        retries: Number of retries on validation failure (default 1).
        base_url: Base URL for the model provider. Used with Ollama models;
            defaults to ``http://localhost:11434`` when *None* and the model
            prefix is ``ollama:``.
    """

    _DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        *,
        model: str,
        output_type: type[T],
        system_prompt: str,
        retries: int = 1,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._output_type = output_type
        self._system_prompt = system_prompt
        self._retries = retries
        self._base_url = base_url

    def _get_agent(self) -> Any:
        """Lazily import pydantic-ai and create an Agent."""
        try:
            from pydantic_ai import Agent
        except ImportError:
            raise ImportError(
                "pydantic-ai is required for LLMParser. "
                "Install it with: pip install 'billfox[llm]'"
            ) from None

        model: Any
        if self._model.startswith("ollama:"):
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider

            ollama_model_name = self._model.split(":", 1)[1]
            base_url = self._base_url or self._DEFAULT_OLLAMA_BASE_URL
            model = OpenAIModel(
                ollama_model_name,
                provider=OpenAIProvider(base_url=f"{base_url}/v1/"),
            )
        else:
            model = self._model

        return Agent(
            model,
            output_type=self._output_type,
            system_prompt=self._system_prompt,
            retries=self._retries,
        )

    async def parse(self, markdown: str) -> T:
        """Parse markdown into a structured Pydantic model instance.

        Args:
            markdown: The markdown text to parse.

        Returns:
            An instance of the configured output_type.
        """
        agent = self._get_agent()
        result = await agent.run(markdown)
        return result.output  # type: ignore[no-any-return]
