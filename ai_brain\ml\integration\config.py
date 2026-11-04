"""
Configuration for enabling, disabling, and tuning AI Brain ML components.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MLConfiguration:
    """
    Granular feature flags for AI Brain Machine Learning components.
    
    Guarantees:
    - If every ML flag is disabled (False), AI Brain operates 100% deterministically
      using existing rule-based logic without degradation.
    """
    enable_importance_ml: bool = True
    enable_ranking_ml: bool = True
    enable_duplicate_ml: bool = True
    enable_conflict_ml: bool = True
    enable_staleness_ml: bool = True
    min_confidence_threshold: float = 0.60
    allow_fallback_on_error: bool = True
