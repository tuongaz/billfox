from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMParser(Generic[T]):
    """Parse markdown into structured data using an LLM.

    Implements the Parser[T] protocol.

    Args:
        model: LLM model identifier (e.g. ``'openai:gpt-4.1'``).
        output_type: A Pydantic BaseModel subclass to parse into.
        system_prompt: System prompt guiding the LLM extraction.
        retries: Number of retries on validation failure (default 1).
    """

    def __init__(
        self,
        *,
        model: str,
        output_type: type[T],
        system_prompt: str,
        retries: int = 1,
    ) -> None:
        self._model = model
        self._output_type = output_type
        self._system_prompt = system_prompt
        self._retries = retries

    def _get_agent(self) -> Any:
        """Lazily import pydantic-ai and create an Agent."""
        try:
            from pydantic_ai import Agent
        except ImportError:
            raise ImportError(
                "pydantic-ai is required for LLMParser. "
                "Install it with: pip install 'billfox[llm]'"
            ) from None
        return Agent(
            self._model,
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
