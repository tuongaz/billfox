# billfox

[![PyPI version](https://img.shields.io/pypi/v/billfox.svg)](https://pypi.org/project/billfox/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/billfox-ai/billfox/actions/workflows/ci.yml/badge.svg)](https://github.com/billfox-ai/billfox/actions)

**Composable document data extraction**: load, preprocess, OCR, LLM parse, store with vector search.

billfox is a Python library that lets you build document processing pipelines from independent, swappable stages. Each stage implements a simple protocol, so you can mix built-in modules with your own.

## Architecture

```
                          billfox pipeline
 ┌─────────┐  ┌──────────────┐  ┌───────────┐  ┌────────┐  ┌───────┐
 │  Source  │→ │ Preprocessor │→ │ Extractor │→ │ Parser │→ │ Store │
 │         │  │   (optional)  │  │   (OCR)   │  │ (LLM)  │  │       │
 └─────────┘  └──────────────┘  └───────────┘  └────────┘  └───────┘
  LocalFile    Resize, YOLO,     MistralOCR     LLMParser   SQLite +
               Chain                            (any LLM)   hybrid
                                                            search
```

**Protocols at every boundary** -- implement `DocumentSource`, `Preprocessor`, `Extractor`, `Parser[T]`, `Embedder`, or `DocumentStore[T]` to plug in your own components.

## Installation

```bash
# Core only (just types and protocols)
pip install billfox

# With Mistral OCR
pip install 'billfox[mistral]'

# With LLM parsing (pydantic-ai)
pip install 'billfox[llm]'

# With SQLite storage and search
pip install 'billfox[store]'

# With CLI
pip install 'billfox[cli]'

# Everything
pip install 'billfox[all]'
```

## Quick Start

### 1. OCR Only -- Extract Markdown from a Document

```python
import asyncio
from billfox.source import LocalFileSource
from billfox.extract import MistralExtractor

async def main():
    source = LocalFileSource()
    extractor = MistralExtractor()  # uses MISTRAL_API_KEY env var

    doc = await source.load("invoice.pdf")
    result = await extractor.extract(doc)
    print(result.markdown)

asyncio.run(main())
```

### 2. Full Pipeline -- OCR + LLM Parse + Store

```python
import asyncio
from pydantic import BaseModel
from billfox import Pipeline
from billfox.source import LocalFileSource
from billfox.extract import MistralExtractor
from billfox.parse import LLMParser
from billfox.preprocess import ResizePreprocessor
from billfox.store import SQLiteDocumentStore

class Invoice(BaseModel):
    vendor_name: str
    total: float
    date: str

async def main():
    pipeline = Pipeline(
        source=LocalFileSource(),
        extractor=MistralExtractor(),
        parser=LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract invoice fields from this document.",
        ),
        preprocessors=[ResizePreprocessor(max_side=1024)],
        store=SQLiteDocumentStore(db_path="invoices.db", schema=Invoice),
    )

    invoice = await pipeline.run("scan.jpg", document_id="inv-001")
    print(f"{invoice.vendor_name}: ${invoice.total}")

asyncio.run(main())
```

### 3. CLI -- Process from the Terminal

```bash
# Extract markdown via OCR
billfox extract receipt.jpg

# Parse into structured JSON
billfox parse receipt.jpg --schema ./models.py:Receipt --model openai:gpt-4.1

# Search stored documents
billfox search "coffee" --db invoices.db

# Configure API keys
billfox config set api_keys.mistral sk-...
```

## Optional Extras

| Extra     | Packages                         | Use case                        |
|-----------|----------------------------------|---------------------------------|
| `mistral` | `mistralai`                      | Mistral OCR extraction          |
| `yolo`    | `onnxruntime`, `numpy`, `Pillow` | YOLO document cropping          |
| `llm`     | `pydantic-ai`                    | LLM structured parsing          |
| `openai`  | `openai`                         | OpenAI text embeddings          |
| `store`   | `sqlalchemy`, `aiosqlite`, `sqlite-vec` | SQLite storage + search  |
| `cli`     | `typer`, `rich`, `tomli-w`       | Command-line interface          |
| `all`     | All of the above                 | Everything                      |

## Documentation

Full documentation is available at [docs/](docs/):

- [Getting Started](docs/getting-started.md)
- [Custom Extractor](docs/custom-extractor.md)
- [Custom Preprocessor](docs/custom-preprocessor.md)
- [Storage and Search](docs/storage-and-search.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running tests, and submitting pull requests.

## License

MIT -- see [LICENSE](LICENSE) for details.
