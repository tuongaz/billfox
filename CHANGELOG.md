# Changelog

## v0.1.0

Initial release of billfox -- composable document data extraction for Python.

### Core
- Frozen dataclasses: `Document`, `Page`, `ExtractionResult`, `SearchResult`
- Protocols: `DocumentSource`, `Preprocessor`, `Extractor`, `Parser[T]`, `Embedder`, `DocumentStore[T]`
- `Pipeline[T]` compositor with `run()` and `extract_only()` methods
- PEP 561 `py.typed` marker for type checker support

### Source
- `LocalFileSource` -- load images and PDFs from the local filesystem

### Preprocessing
- `ResizePreprocessor` -- resize images maintaining aspect ratio
- `YOLOPreprocessor` -- crop documents using YOLO ONNX object detection
- `PreprocessorChain` -- chain multiple preprocessors sequentially

### Extraction
- `MistralExtractor` -- OCR via Mistral AI API (images and PDFs)

### Parsing
- `LLMParser[T]` -- parse markdown into any Pydantic model using pydantic-ai

### Embedding
- `OpenAIEmbedder` -- generate text embeddings via OpenAI API
- `encode_vector` / `decode_vector` -- base64 float packing utilities

### Storage
- `SQLiteDocumentStore[T]` -- SQLite-backed storage with Pydantic serialization
- Hybrid search: BM25 (FTS5) + vector KNN (sqlite-vec) with RRF fusion
- Three search modes: hybrid, bm25, vector

### CLI
- `billfox extract` -- OCR a document to markdown
- `billfox parse` -- full pipeline to structured JSON
- `billfox search` -- query stored documents
- `billfox config` -- manage API keys and settings
