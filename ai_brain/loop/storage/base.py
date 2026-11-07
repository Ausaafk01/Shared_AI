"""
Abstract base class defining the storage interface for Brain Loop interactions.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from ai_brain.loop.models import InteractionRecord, InteractionStatus


class InteractionStore(ABC):
    """
    Abstract interface for persisting and querying Brain Loop interaction records.
    
    Guarantees:
    - Interaction state survives application and process restarts.
    - Preserves links between input, context, response, and audit events.
    - Thread-safe persistence and querying.
    """

    @abstractmethod
    def save(self, record: InteractionRecord) -> None:
        """Persist or update an InteractionRecord."""
        pass

    @abstractmethod
    def get(self, interaction_id: str, project_id: Optional[str] = None) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by its unique interaction_id."""
        pass

    @abstractmethod
    def get_by_idempotency_key(self, idempotency_key: str, project_id: Optional[str] = None) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by its idempotency_key."""
        pass

    @abstractmethod
    def list(
        self,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[Union[str, InteractionStatus]] = None,
    ) -> List[InteractionRecord]:
        """List and filter interaction records."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close underlying storage connections."""
        pass

