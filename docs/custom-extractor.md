# Custom Extractor

An extractor converts a `Document` (raw bytes) into an `ExtractionResult` (markdown text). billfox ships with `MistralExtractor`, but you can implement your own.

## The Extractor Protocol

```python
from billfox._types import Document, ExtractionResult

class Extractor(Protocol):
    async def extract(self, document: Document) -> ExtractionResult: ...
```

## Example: Tesseract OCR Extractor

```python
import asyncio
import subprocess
import tempfile
from pathlib import Path
from billfox._types import Document, ExtractionResult, Page

class TesseractExtractor:
    """Extract text using Tesseract OCR."""

    def __init__(self, lang: str = "eng") -> None:
        self._lang = lang

    async def extract(self, document: Document) -> ExtractionResult:
        return await asyncio.to_thread(self._extract_sync, document)

    def _extract_sync(self, document: Document) -> ExtractionResult:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(document.content)
            tmp.flush()
            result = subprocess.run(
                ["tesseract", tmp.name, "stdout", "-l", self._lang],
                capture_output=True, text=True, check=True,
            )
        markdown = result.stdout.strip()
        return ExtractionResult(
            markdown=markdown,
            pages=[Page(index=0, markdown=markdown)],
        )
```

## Using Your Extractor

```python
from billfox import Pipeline
from billfox.source import LocalFileSource

pipeline = Pipeline(
    source=LocalFileSource(),
    extractor=TesseractExtractor(lang="eng"),
    parser=my_parser,
)
```

## Tips

- Use `asyncio.to_thread()` to wrap synchronous SDK calls so your extractor works with the async pipeline.
- Return one `Page` per logical page of the document. For single-page images, one page is fine.
- The `ExtractionResult.metadata` dict is a good place to store extractor-specific info (e.g. confidence scores, model version).
- Your extractor is automatically compatible with `Pipeline` and `extract_only()` without any registration.
