"""
Abstract storage interface for persistent session records.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ai_brain.agents.sessions.models import SessionRecord, SessionStatus


class SessionStore(ABC):
    """
    Abstract storage interface for session records.
    
    Guarantees:
    - Session continuity and state survive application/process restarts.
    - Thread-safe persistence and querying.
    """

    @abstractmethod
    def save(self, session: SessionRecord) -> None:
        """Persist or update a SessionRecord."""
        pass

    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by its unique session_id."""
        pass

    @abstractmethod
    def list(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
    ) -> List[SessionRecord]:
        """List and filter session records."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session record."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connections."""
        pass

