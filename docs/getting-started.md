# Getting Started

## Installation

```bash
pip install 'billfox[all]'
```

Or install only what you need:

```bash
pip install 'billfox[mistral,llm]'   # OCR + LLM parsing
pip install 'billfox[store]'          # SQLite storage and search
pip install 'billfox[cli]'            # Command-line interface
```

## Your First Pipeline

Define a Pydantic model for the data you want to extract, then wire up a pipeline:

```python
import asyncio
from pydantic import BaseModel
from billfox import Pipeline
from billfox.source import LocalFileSource
from billfox.extract import MistralExtractor
from billfox.parse import LLMParser

class Receipt(BaseModel):
    vendor: str
    total: float
    date: str
    items: list[str] = []

async def main():
    pipeline = Pipeline(
        source=LocalFileSource(),
        extractor=MistralExtractor(),  # uses MISTRAL_API_KEY env var
        parser=LLMParser(
            model="openai:gpt-4.1",
            output_type=Receipt,
            system_prompt="Extract receipt fields from this document.",
        ),
    )

    receipt = await pipeline.run("receipt.jpg")
    print(f"Vendor: {receipt.vendor}")
    print(f"Total: ${receipt.total}")

asyncio.run(main())
```

## Extract Only (No LLM)

If you just need the OCR markdown without LLM parsing:

```python
import asyncio
from billfox.source import LocalFileSource
from billfox.extract import MistralExtractor

async def main():
    source = LocalFileSource()
    extractor = MistralExtractor()

    doc = await source.load("document.pdf")
    result = await extractor.extract(doc)
    print(result.markdown)

asyncio.run(main())
```

## Adding Preprocessing

Chain preprocessors to improve extraction quality:

```python
from billfox.preprocess import ResizePreprocessor, PreprocessorChain, YOLOPreprocessor

pipeline = Pipeline(
    source=LocalFileSource(),
    extractor=MistralExtractor(),
    parser=parser,
    preprocessors=[
        ResizePreprocessor(max_side=1024),
    ],
)
```

For YOLO-based document cropping (requires an ONNX model file):

```python
pipeline = Pipeline(
    source=LocalFileSource(),
    extractor=MistralExtractor(),
    parser=parser,
    preprocessors=[
        YOLOPreprocessor(model_path="model.onnx"),
        ResizePreprocessor(max_side=1024),
    ],
)
```

## Using the CLI

```bash
# Set up API keys
billfox config set api_keys.mistral sk-...

# Extract markdown from a document
billfox extract invoice.pdf

# Parse into structured JSON
billfox parse invoice.pdf --schema ./models.py:Invoice --output result.json

# Search stored documents
billfox search "coffee shop" --db receipts.db
```

## Core Types

billfox uses frozen dataclasses for its core types:

- **`Document`** -- loaded file with `content` (bytes), `mime_type`, `source_uri`, and `metadata`
- **`ExtractionResult`** -- OCR output with `markdown`, `pages` (list of `Page`), and `metadata`
- **`SearchResult`** -- search hit with `document_id`, `data`, `score`, and `signals` (scoring breakdown)
