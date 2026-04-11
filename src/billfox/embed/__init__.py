"""Text embedding modules."""

from billfox.embed._base import Embedder
from billfox.embed.openai import OpenAIEmbedder, decode_vector, encode_vector

__all__ = ["Embedder", "OpenAIEmbedder", "decode_vector", "encode_vector"]
