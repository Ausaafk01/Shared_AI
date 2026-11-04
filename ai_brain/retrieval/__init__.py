"""
Retrieval Engine subsystem for AI Brain.
"""

from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.exceptions import (
    GraphRetrievalError,
    HybridFusionError,
    ProjectIsolationError,
    QueryValidationError,
    RetrievalError,
    RetrievalExecutionError,
    ScoringError,
    VectorRetrievalError,
)
from ai_brain.retrieval.fusion import FusedCandidate, HybridFusion
from ai_brain.retrieval.graph_retriever import GraphMatch, GraphRetriever
from ai_brain.retrieval.models import (
    ProcessedQuery,
    QueryFilters,
    RetrievalItem,
    RetrievalResponse,
    ScoreBreakdown,
)
from ai_brain.retrieval.query import QueryProcessor
from ai_brain.retrieval.scoring import RelevanceScorer
from ai_brain.retrieval.types import (
    RetrievalMethod,
    SeedStrategy,
    normalize_retrieval_method,
    normalize_seed_strategy,
)
from ai_brain.retrieval.vector_retriever import VectorRetriever

__all__ = [
    # Engine
    "RetrievalEngine",
    # Types
    "RetrievalMethod",
    "SeedStrategy",
    "normalize_retrieval_method",
    "normalize_seed_strategy",
    # Models
    "QueryFilters",
    "ProcessedQuery",
    "ScoreBreakdown",
    "RetrievalItem",
    "RetrievalResponse",
    # Components
    "QueryProcessor",
    "GraphRetriever",
    "GraphMatch",
    "VectorRetriever",
    "HybridFusion",
    "FusedCandidate",
    "RelevanceScorer",
    # Exceptions
    "RetrievalError",
    "QueryValidationError",
    "ProjectIsolationError",
    "RetrievalExecutionError",
    "GraphRetrievalError",
    "VectorRetrievalError",
    "HybridFusionError",
    "ScoringError",
]

