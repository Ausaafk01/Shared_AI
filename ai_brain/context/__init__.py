"""
Context Builder subsystem for AI Brain.

Converts retrieved memories into a compact, prioritized, token-budget-aware
context package that can be supplied to an AI agent.
"""

from ai_brain.context.budget import TokenBudgetManager
from ai_brain.context.builder import ContextBuilder
from ai_brain.context.compressor import ContextCompressor
from ai_brain.context.exceptions import (
    CompressionError,
    ContextBuilderError,
    ContextError,
    ContextProjectIsolationError,
    FormattingError,
    PrioritizationError,
    SelectionError,
    TokenBudgetExceededError,
    TokenizerError,
)
from ai_brain.context.formatter import ContextFormatter
from ai_brain.context.models import (
    ContextItem,
    ContextPackage,
    ContextSection,
    ExcludedItem,
    PriorityBreakdown,
)
from ai_brain.context.prioritizer import MemoryPrioritizer
from ai_brain.context.selector import MemorySelector, SelectionCriteria
from ai_brain.context.tokenizer import BaseTokenizer, DeterministicTokenizer

__all__ = [
    "ContextError",
    "ContextBuilderError",
    "TokenBudgetExceededError",
    "SelectionError",
    "PrioritizationError",
    "CompressionError",
    "FormattingError",
    "TokenizerError",
    "ContextProjectIsolationError",
    "BaseTokenizer",
    "DeterministicTokenizer",
    "PriorityBreakdown",
    "ContextItem",
    "ExcludedItem",
    "ContextSection",
    "ContextPackage",
    "SelectionCriteria",
    "MemorySelector",
    "MemoryPrioritizer",
    "ContextCompressor",
    "ContextFormatter",
    "TokenBudgetManager",
    "ContextBuilder",
]

