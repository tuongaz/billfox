# billfox

**Composable document data extraction**: load, preprocess, OCR, LLM parse, store with vector search.

billfox is a Python library for building document processing pipelines from independent, swappable stages. Each stage implements a simple protocol, so you can mix built-in modules with your own.

## Architecture

```
 ┌─────────┐  ┌──────────────┐  ┌───────────┐  ┌────────┐  ┌───────┐
 │  Source  │→ │ Preprocessor │→ │ Extractor │→ │ Parser │→ │ Store │
 └─────────┘  └──────────────┘  └───────────┘  └────────┘  └───────┘
```

Each stage is defined by a **protocol** (Python `Protocol` class). Implement the protocol to create your own components, or use the built-in ones.

| Stage        | Protocol         | Built-in                    |
|--------------|------------------|-----------------------------|
| Source       | `DocumentSource` | `LocalFileSource`           |
| Preprocessor | `Preprocessor`   | `ResizePreprocessor`, `YOLOPreprocessor`, `PreprocessorChain` |
| Extractor    | `Extractor`      | `MistralExtractor`          |
| Parser       | `Parser[T]`      | `LLMParser[T]`              |
| Embedder     | `Embedder`       | `OpenAIEmbedder`            |
| Store        | `DocumentStore[T]` | `SQLiteDocumentStore[T]`  |

## Next Steps

- [Getting Started](getting-started.md) -- install and run your first pipeline
- [Custom Extractor](custom-extractor.md) -- implement your own OCR/extraction
- [Custom Preprocessor](custom-preprocessor.md) -- add image preprocessing steps
- [Storage and Search](storage-and-search.md) -- persist and query documents
