"""
Abstract storage interface for persistent Agent Registry records.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ai_brain.agents.models import AgentRecord, AgentStatus, AgentType


class AgentRegistryStore(ABC):
    """
    Abstract interface for persisting registered agent identities.
    
    Guarantees:
    - Agent identity and configuration survive application/process restarts.
    - Thread-safe persistence and querying.
    """

    @abstractmethod
    def save(self, record: AgentRecord) -> None:
        """Persist or update an AgentRecord."""
        pass

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an AgentRecord by its agent_id."""
        pass

    @abstractmethod
    def list(
        self,
        *,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        provider: Optional[str] = None,
    ) -> List[AgentRecord]:
        """List and filter registered AgentRecords."""
        pass

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Remove an AgentRecord from persistent storage."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close underlying storage connections."""
        pass

