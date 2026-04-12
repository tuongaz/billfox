from __future__ import annotations

import mimetypes
from pathlib import Path

from billfox._types import Document

_SUPPORTED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
        "image/heic",
        "image/bmp",
        "application/pdf",
    }
)

# Register MIME types not always present in the default database.
mimetypes.add_type("image/heic", ".heic")
mimetypes.add_type("image/heic", ".HEIC")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/bmp", ".bmp")


class LocalFileSource:
    """Load documents from the local filesystem."""

    async def load(self, uri: str) -> Document:
        """Read a file and return a ``Document``.

        Args:
            uri: Filesystem path to the file.

        Returns:
            A ``Document`` populated with the file's bytes and detected MIME type.

        Raises:
            FileNotFoundError: If *uri* does not point to an existing file.
            ValueError: If the file extension maps to an unsupported MIME type.
        """
        path = Path(uri)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {uri}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None or mime_type not in _SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported file type: {path.suffix!r}. "
                f"Supported MIME types: {', '.join(sorted(_SUPPORTED_MIME_TYPES))}"
            )

        content = path.read_bytes()

        return Document(
            content=content,
            mime_type=mime_type,
            source_uri=str(path),
            metadata={},
        )
