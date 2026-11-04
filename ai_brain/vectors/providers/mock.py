"""
Deterministic mock embedding provider for tests and offline usage.
"""

import hashlib
import math
import re
from typing import List, Sequence

from ai_brain.vectors.exceptions import EmbeddingProviderError
from ai_brain.vectors.providers.base import EmbeddingProvider


SEMANTIC_CLUSTERS = {
    "concept:auth": {"auth", "authentication", "login", "log", "jwt", "tokens", "token", "password", "session", "credentials"},
    "concept:database": {"database", "databases", "db", "sql", "postgres", "postgresql", "mysql", "sqlite", "relational", "storage", "store", "stores", "persistence", "persistent", "query"},
    "concept:caching": {"cache", "caching", "redis", "memcached", "in-memory"},
    "concept:preference": {"prefer", "prefers", "preference", "concise", "style", "formatting", "conventions", "functional"},
    "concept:testing": {"test", "tests", "testing", "pytest", "unittest", "coverage"},
    "concept:security": {"vault", "secret", "secrets", "key", "keys", "crypto", "encryption"},
    "concept:network": {"network", "port", "http", "https", "server", "endpoint", "api"},
}


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, offline embedding provider generating reproducible semantic-like vectors.
    
    Guarantees:
    - Zero external network calls or binary downloads.
    - Deterministic: identical texts always yield byte-for-byte identical vectors.
    - L2-normalized: all output vectors have unit norm (||v||_2 = 1.0).
    - Semantic similarity: uses tokenization, subwords, and frequency hashing so
      semantically related texts (e.g. 'PostgreSQL database' and 'Postgres relational store')
      yield high cosine similarity, while unrelated texts yield low similarity.
    - Configurable dimension (default 384, matching all-MiniLM-L6-v2) and model_id.
    - Configurable simulate_failure mode for error-handling verification.
    """

    def __init__(
        self,
        dimension: int = 384,
        model_id: str = "mock-semantic-v1",
        *,
        simulate_failure: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        self._dimension = dimension
        self._model_id = model_id
        self.simulate_failure = simulate_failure

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized float vector for text.
        """
        if self.simulate_failure:
            raise EmbeddingProviderError(f"Simulated provider failure for model '{self.model_id}'")

        cleaned = self.validate_input_text(text)

        # Initialize raw dense vector
        raw_vec = [0.0] * self._dimension

        # Tokenize text into words and subword character n-grams with weights
        words = re.findall(r"\w+", cleaned.lower())
        weighted_tokens: List[tuple[str, float]] = []

        # Whole words: weight 2.0
        for w in words:
            weighted_tokens.append((f"word:{w}", 2.0))
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    weighted_tokens.append((f"sub:{w[i:i+3]}", 0.2))

        # Semantic concept mapping to bridge vocabulary mismatches: weight 5.0
        for concept, kw_set in SEMANTIC_CLUSTERS.items():
            if any(w in kw_set for w in words):
                weighted_tokens.append((concept, 5.0))

        if not weighted_tokens:
            weighted_tokens = [(f"word:{cleaned}", 1.0)]

        # Hash each token into the dimension space
        for token_str, token_weight in weighted_tokens:
            h = hashlib.sha256(token_str.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], byteorder="big") % self._dimension
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            magnitude = (1.0 + (h[5] / 255.0)) * token_weight

            raw_vec[bucket] += sign * magnitude

            # Secondary spillover bucket for smoother representation
            secondary_bucket = (bucket + int.from_bytes(h[6:8], byteorder="big")) % self._dimension
            raw_vec[secondary_bucket] += (sign * magnitude) * 0.35

        # Compute L2 norm
        norm = math.sqrt(sum(x * x for x in raw_vec))
        if norm > 1e-12:
            normalized = [round(x / norm, 6) for x in raw_vec]
        else:
            # Fallback uniform unit vector if all tokens cancelled out
            val = round(1.0 / math.sqrt(self._dimension), 6)
            normalized = [val] * self._dimension

        return self.validate_embedding_vector(normalized)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate normalized vectors for multiple texts."""
        if self.simulate_failure:
            raise EmbeddingProviderError(f"Simulated provider failure for model '{self.model_id}'")
        return [self.embed_text(t) for t in texts]
