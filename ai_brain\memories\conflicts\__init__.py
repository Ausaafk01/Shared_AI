"""
Conflict Management package for AI Brain.
"""

from .detector import ConflictDetector
from .evaluator import ConflictAssessment, ConflictEvaluator
from .manager import ConflictManager, ConflictResolutionResult
from .models import MemoryConflict
from .types import (
    ConflictStatus,
    ConflictType,
    is_valid_conflict_type,
    normalize_conflict_status,
    normalize_conflict_type,
    register_conflict_type,
    unregister_conflict_type,
    validate_status_transition,
)

__all__ = [
    "MemoryConflict",
    "ConflictType",
    "ConflictStatus",
    "register_conflict_type",
    "unregister_conflict_type",
    "is_valid_conflict_type",
    "normalize_conflict_type",
    "normalize_conflict_status",
    "validate_status_transition",
    "ConflictDetector",
    "ConflictEvaluator",
    "ConflictAssessment",
    "ConflictManager",
    "ConflictResolutionResult",
]

