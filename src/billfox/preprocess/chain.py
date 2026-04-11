"""Chain multiple preprocessors into a single sequential pipeline."""

from __future__ import annotations

from billfox._types import Document
from billfox.preprocess._base import Preprocessor


class PreprocessorChain:
    """Apply a sequence of preprocessors in order.

    Matches the Preprocessor protocol so it can be used anywhere
    a single preprocessor is expected.
    """

    def __init__(self, preprocessors: list[Preprocessor]) -> None:
        self._preprocessors = list(preprocessors)

    async def process(self, document: Document) -> Document:
        """Apply each preprocessor sequentially, passing the result forward."""
        result = document
        for preprocessor in self._preprocessors:
            result = await preprocessor.process(result)
        return result
