"""
Optional local embedding provider leveraging HuggingFace sentence-transformers.
"""

from typing import Any, List, Optional, Sequence

from ai_brain.vectors.exceptions import EmbeddingProviderError
from ai_brain.vectors.providers.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider backed by sentence-transformers (e.g. all-MiniLM-L6-v2).
    
    Requires optional dependency: `pip install sentence-transformers`
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingProviderError(
                "sentence-transformers is not installed. Install it via: pip install sentence-transformers"
            ) from e

        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
            # Infer dimension from model output
            sample = self._model.encode("test", normalize_embeddings=True)
            self._dimension = len(sample)
            return self._model
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to load embedding model '{self._model_name}': {e}") from e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    @property
    def model_id(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def embed_text(self, text: str) -> List[float]:
        cleaned = self.validate_input_text(text)
        model = self._load_model()
        try:
            emb = model.encode(cleaned, normalize_embeddings=True)
            return self.validate_embedding_vector(emb.tolist())
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to encode text with model '{self.model_id}': {e}") from e

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned_texts = [self.validate_input_text(t) for t in texts]
        model = self._load_model()
        try:
            embs = model.encode(cleaned_texts, normalize_embeddings=True)
            return [self.validate_embedding_vector(emb.tolist()) for emb in embs]
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to encode batch with model '{self.model_id}': {e}") from e

