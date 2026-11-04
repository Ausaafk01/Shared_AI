"""
Exception hierarchy for the AI Brain Vector Memory component.
"""


class VectorError(Exception):
    """Base exception for all Vector Memory operations."""


class EmbeddingError(VectorError):
    """Base exception for embedding generation failures."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails to generate an embedding."""


class VectorValidationError(VectorError):
    """Raised when vector records, inputs, or parameters fail validation."""


class DimensionMismatchError(VectorValidationError):
    """Raised when an embedding dimension does not match the expected index/model dimension."""


class VectorNotFoundError(VectorError):
    """Raised when a requested vector record cannot be found in storage."""


class DuplicateVectorError(VectorError):
    """Raised when attempting to add an identical vector record for the same memory version and model."""


class VectorStorageError(VectorError):
    """Raised when persistent vector storage operations fail."""


class ProjectIsolationError(VectorError):
    """Raised when vector operations attempt to violate project boundaries."""

