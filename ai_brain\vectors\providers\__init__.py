"""
Embedding provider package.
"""

from .base import EmbeddingProvider
from .local import SentenceTransformerEmbeddingProvider
from .mock import MockEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]

