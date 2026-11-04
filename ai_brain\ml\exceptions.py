"""
Exceptions for AI Brain Machine Learning layer.
"""


class MLError(Exception):
    """Base exception for all ML-related errors."""
    pass


class ModelNotTrainedError(MLError):
    """Raised when inference is attempted on an untrained model."""
    pass


class ModelNotFoundError(MLError):
    """Raised when a requested model or version is not found in registry."""
    pass


class FeatureExtractionError(MLError):
    """Raised when feature extraction fails on an input."""
    pass


class DatasetValidationError(MLError):
    """Raised when training data schema or integrity validation fails."""
    pass


class FeedbackValidationError(MLError):
    """Raised when feedback input validation fails."""
    pass


class ModelSerializationError(MLError):
    """Raised when serializing or deserializing a model fails."""
    pass


class FallbackTriggeredWarning(UserWarning):
    """Warning emitted when ML prediction falls back to deterministic logic."""
    pass
