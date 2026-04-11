"""Document source modules for loading files."""

from billfox.source._base import DocumentSource
from billfox.source.local import LocalFileSource

__all__ = ["DocumentSource", "LocalFileSource"]
