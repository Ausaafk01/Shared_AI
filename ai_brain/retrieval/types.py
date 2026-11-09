"""
Enumerations and types for the Retrieval Engine subsystem.
"""

from enum import Enum
from typing import Union


class RetrievalMethod(str, Enum):
    """Supported retrieval methodologies."""
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"


class SeedStrategy(str, Enum):
    """Seed selection strategies for graph retrieval."""
    EXPLICIT = "EXPLICIT"            # Use explicitly provided seed_memory_ids
    KEYWORD_ENTITY = "KEYWORD_ENTITY" # Search graph for nodes matching query keywords/entities
    VECTOR_TOP_K = "VECTOR_TOP_K"     # Use top vector search matches as graph traversal seeds
    HYBRID_SEED = "HYBRID_SEED"       # Explicit if given, else vector seeds if available, else keywords


def normalize_retrieval_method(method: Union[str, RetrievalMethod]) -> RetrievalMethod:
    """Normalize and validate a retrieval method input."""
    if isinstance(method, RetrievalMethod):
        return method
    if isinstance(method, str):
        val = method.strip().upper()
        try:
            return RetrievalMethod(val)
        except ValueError:
            valid = [m.value for m in RetrievalMethod]
            raise ValueError(f"Invalid retrieval method '{method}'. Must be one of: {valid}")
    raise ValueError(f"Expected str or RetrievalMethod, got {type(method).__name__}")


def normalize_seed_strategy(strategy: Union[str, SeedStrategy]) -> SeedStrategy:
    """Normalize and validate a seed selection strategy."""
    if isinstance(strategy, SeedStrategy):
        return strategy
    if isinstance(strategy, str):
        val = strategy.strip().upper()
        try:
            return SeedStrategy(val)
        except ValueError:
            valid = [s.value for s in SeedStrategy]
            raise ValueError(f"Invalid seed strategy '{strategy}'. Must be one of: {valid}")
    raise ValueError(f"Expected str or SeedStrategy, got {type(strategy).__name__}")

