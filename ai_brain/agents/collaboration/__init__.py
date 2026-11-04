"""
Multi-agent collaboration package for AI Brain.
"""

from .models import (
    CollaborationStep,
    CollaborationStepResult,
    CollaborationWorkflowRecord,
    WorkflowStatus,
)
from .workflow import CollaborationEngine

__all__ = [
    "CollaborationStep",
    "CollaborationStepResult",
    "CollaborationWorkflowRecord",
    "WorkflowStatus",
    "CollaborationEngine",
]

