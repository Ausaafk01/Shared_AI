"""Abstract base class defining the persistent storage contract for Memories."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ai_brain.memories.models import Memory


class MemoryStore(ABC):
    """Abstract interface defining required persistent memory storage operations."""

    @abstractmethod
    def store(self, memory: Memory) -> None:
        """Persist a memory instance."""
        pass

    def save(self, memory: Memory) -> None:
        """Convenience alias for store."""
        self.store(memory)

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory snapshot by its unique memory_id."""
        pass

    @abstractmethod
    def get_by_root_id(self, root_memory_id: str) -> List[Memory]:
        """Retrieve all versions in a conceptual memory's version chain, ordered by version asc."""
        pass

    @abstractmethod
    def get_current(self, root_memory_id: str) -> Optional[Memory]:
        """Retrieve the current active (or latest) version for a given root_memory_id."""
        pass

    @abstractmethod
    def list(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """List and filter memories by project, status, and/or tier."""
        pass

    @abstractmethod
    def update_status(self, memory_id: str, new_status: str) -> bool:
        """Update the lifecycle status (e.g. ACTIVE -> SUPERSEDED, DISPUTED) of a memory."""
        pass

    @abstractmethod
    def count(self, *, project_id: Optional[str] = None, status: Optional[str] = None) -> int:
        """Count memories matching the given filter criteria."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close underlying storage resources. Idempotent."""
        pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
