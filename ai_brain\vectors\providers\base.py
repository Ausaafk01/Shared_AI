"""
Abstract base class and contract for embedding providers.
"""

from abc import ABC, abstractmethod
import math
from typing import List, Sequence

from ai_brain.vectors.exceptions import DimensionMismatchError, VectorValidationError


class EmbeddingProvider(ABC):
    """
    Abstract interface for generating vector embeddings from text.
    
    All providers must generate fixed-dimension float vectors and identify
    their model via a stable model_id.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension produced by this provider."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier of the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized embedding vector for a single text string.
        
        Args:
            text: Non-empty text string.
            
        Returns:
            List of floats representing the embedding vector.
        """
        pass

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.
        
        Args:
            texts: Sequence of non-empty text strings.
            
        Returns:
            List of embedding vectors corresponding to the input texts.
        """
        return [self.embed_text(t) for t in texts]

    def validate_input_text(self, text: str) -> str:
        """
        Validate input text before embedding generation.
        
        Raises:
            VectorValidationError: If text is empty, None, or whitespace-only.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            raise VectorValidationError("Input text for embedding must be a non-empty string")
        return text.strip()

    def validate_embedding_vector(self, vector: Sequence[float]) -> List[float]:
        """
        Validate generated embedding vector.
        
        Raises:
            VectorValidationError: If vector is malformed, empty, or contains non-numeric/NaN/Inf values.
            DimensionMismatchError: If vector dimension does not match provider.dimension.
        """
        if not vector or not isinstance(vector, (list, tuple)):
            raise VectorValidationError("Embedding vector must be a non-empty list or tuple of floats")

        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        validated: List[float] = []
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise VectorValidationError(f"Vector component at index {i} is not a valid number: {val!r}")
            val_float = float(val)
            if math.isnan(val_float) or math.isinf(val_float):
                raise VectorValidationError(f"Vector component at index {i} cannot be NaN or Infinite")
            validated.append(val_float)

        return validated

