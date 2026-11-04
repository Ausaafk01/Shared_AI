"""
AI Brain Vector Memory subsystem.
"""

from .exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    EmbeddingError,
    EmbeddingProviderError,
    ProjectIsolationError,
    VectorError,
    VectorNotFoundError,
    VectorStorageError,
    VectorValidationError,
)
from .index import VectorIndex, compute_cosine_similarity
from .manager import VectorMemoryManager
from .models import VectorMatch, VectorRecord, VectorSearchResult
from .providers import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from .storage import SQLiteVectorStore, VectorStore

__all__ = [
    # Models
    "VectorRecord",
    "VectorMatch",
    "VectorSearchResult",
    # Providers
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    # Storage
    "VectorStore",
    "SQLiteVectorStore",
    # Index & Search
    "VectorIndex",
    "compute_cosine_similarity",
    # Manager
    "VectorMemoryManager",
    # Exceptions
    "VectorError",
    "EmbeddingError",
    "EmbeddingProviderError",
    "VectorValidationError",
    "DimensionMismatchError",
    "VectorNotFoundError",
    "DuplicateVectorError",
    "VectorStorageError",
    "ProjectIsolationError",
]

