# Custom Preprocessor

A preprocessor transforms a `Document` before extraction. Common uses include resizing images, cropping regions of interest, or converting formats. billfox ships with `ResizePreprocessor`, `YOLOPreprocessor`, and `PreprocessorChain`.

## The Preprocessor Protocol

```python
from billfox._types import Document

class Preprocessor(Protocol):
    async def process(self, document: Document) -> Document: ...
```

The protocol is simple: take a `Document`, return a (possibly modified) `Document`.

## Example: Grayscale Converter

```python
import io
from PIL import Image
from billfox._types import Document

class GrayscalePreprocessor:
    """Convert colour images to grayscale for better OCR accuracy."""

    _IMAGE_TYPES = frozenset({
        "image/jpeg", "image/png", "image/webp",
    })

    async def process(self, document: Document) -> Document:
        if document.mime_type not in self._IMAGE_TYPES:
            return document

        img = Image.open(io.BytesIO(document.content)).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return Document(
            content=buf.getvalue(),
            mime_type="image/png",
            source_uri=document.source_uri,
            metadata={**document.metadata, "preprocessor": "grayscale"},
        )
```

## Chaining Preprocessors

Use `PreprocessorChain` to apply multiple preprocessors in sequence:

```python
from billfox.preprocess import PreprocessorChain, ResizePreprocessor

chain = PreprocessorChain([
    GrayscalePreprocessor(),
    ResizePreprocessor(max_side=1024),
])
```

Or pass them as a list to the pipeline (they are applied in order):

```python
pipeline = Pipeline(
    source=source,
    extractor=extractor,
    parser=parser,
    preprocessors=[
        GrayscalePreprocessor(),
        ResizePreprocessor(max_side=1024),
    ],
)
```

## Tips

- Non-image documents (e.g. PDFs) should pass through unchanged -- check `document.mime_type` before processing.
- `Document` is a frozen dataclass, so return a new instance rather than mutating.
- Add a `"preprocessor"` key to metadata to record which preprocessor was applied -- useful for debugging.
- `PreprocessorChain` itself satisfies the `Preprocessor` protocol, so chains can be nested.
