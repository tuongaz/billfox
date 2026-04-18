"""Text embedding modules."""

from billfox.embed._base import Embedder
from billfox.embed.ollama import OllamaEmbedder
from billfox.embed.openai import OpenAIEmbedder, decode_vector, encode_vector

__all__ = ["Embedder", "OllamaEmbedder", "OpenAIEmbedder", "decode_vector", "encode_vector"]
