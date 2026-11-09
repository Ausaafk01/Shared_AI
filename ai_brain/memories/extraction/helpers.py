"""
Helper structures and parsing utilities for memory extraction.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Mapping, Optional, Sequence

from ai_brain.events.models import Event
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType


# Words/phrases indicating speculation or uncertainty (not established facts)
SPECULATION_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bi think\b",
    r"\bi guess\b",
    r"\bpossibly\b",
    r"\bassuming\b",
    r"\bsuppose\b",
    r"\bnot sure\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bwe could\b",
]

# Words/phrases indicating transient or temporary preferences/states
TRANSIENT_PATTERNS = [
    r"\bfor now\b",
    r"\bright now\b",
    r"\btemporarily\b",
    r"\bjust this once\b",
    r"\bat the moment\b",
    r"\bjust for testing\b",
]

# Casual conversation phrases that contain no lasting knowledge
CASUAL_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|good morning|good evening)[\.!]?$",
    r"^(sounds good|looks good|got it|sure thing|cool|nice)[\.!]?$",
]


def is_speculation(text: str) -> bool:
    """Return True if the text expresses speculation, uncertainty, or unverified assumptions."""
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in SPECULATION_PATTERNS)


def is_transient(text: str) -> bool:
    """Return True if the text indicates a temporary or transient state/preference."""
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in TRANSIENT_PATTERNS)


def is_casual_noise(text: str) -> bool:
    """Return True if the text is casual conversation with no lasting information."""
    text_clean = text.strip().lower()
    return any(re.match(pat, text_clean) for pat in CASUAL_PATTERNS)


def extract_text_from_data(data: Any, preferred_keys: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    Extract meaningful string content from an Event payload (string, dict, or list).
    
    Args:
        data: Arbitrary payload from event.input, event.output, or event.result.
        preferred_keys: Specific dictionary keys to prioritize.
        
    Returns:
        Extracted string or None.
    """
    if data is None:
        return None
    if isinstance(data, str):
        clean = data.strip()
        return clean if clean else None
    
    if isinstance(data, (dict, Mapping)):
        keys = list(preferred_keys or []) + [
            "message", "text", "content", "summary", "prompt", "query",
            "decision", "reason", "error", "command", "response", "plan",
        ]
        for key in keys:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key].strip()
        
        # If no standard string key matches, check first string value
        for val in data.values():
            if isinstance(val, str) and val.strip():
                return val.strip()

    return None


@dataclass
class CandidateMemory:
    """
    Intermediate representation of a candidate memory before construction and deduplication.
    """
    memory_type: str | MemoryType
    content: str
    project_id: str
    source_event_ids: list[str] = field(default_factory=list)
    source_agent_ids: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 1.0
    status: str = MemoryStatus.ACTIVE.value
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    relationships: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Memory:
        """Construct an immutable validated Memory object."""
        return Memory.create(
            memory_type=self.memory_type,
            content=self.content,
            project_id=self.project_id,
            source_event_ids=self.source_event_ids,
            source_agent_ids=self.source_agent_ids,
            importance=self.importance,
            confidence=self.confidence,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            relationships=self.relationships,
            metadata=self.metadata,
        )

