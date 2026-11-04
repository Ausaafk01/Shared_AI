"""
Abstract base class for vector record storage.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ai_brain.vectors.models import VectorRecord


class VectorStore(ABC):
    """
    Abstract interface for persistent and in-memory vector storage backends.
    """

    @abstractmethod
    def add_vector(self, record: VectorRecord, *, fail_if_exists: bool = False) -> VectorRecord:
        """
        Store a VectorRecord.
        
        Args:
            record: The VectorRecord to store.
            fail_if_exists: If True, raise DuplicateVectorError when duplicate exists.
            
        Returns:
            The stored or existing VectorRecord.
        """
        pass

    @abstractmethod
    def get_vector(self, record_id: str) -> Optional[VectorRecord]:
        """Retrieve a vector record by record_id."""
        pass

    @abstractmethod
    def get_vector_by_memory(
        self,
        project_id: str,
        memory_id: str,
        version: int,
        model_id: str,
    ) -> Optional[VectorRecord]:
        """Retrieve a vector record by project, memory_id, version, and model_id."""
        pass

    @abstractmethod
    def update_vector(self, record: VectorRecord) -> VectorRecord:
        """Update an existing vector record in storage."""
        pass

    @abstractmethod
    def delete_vector(self, record_id: str) -> bool:
        """Delete a vector record by record_id. Returns True if deleted."""
        pass

    @abstractmethod
    def delete_vectors_by_memory(self, project_id: str, memory_id: str) -> int:
        """Delete all vector records associated with a memory_id. Returns count deleted."""
        pass

    @abstractmethod
    def get_vectors_by_project(
        self,
        project_id: str,
        *,
        model_id: Optional[str] = None,
        allowed_statuses: Optional[Sequence[str]] = None,
    ) -> List[VectorRecord]:
        """Retrieve all vector records for a project, optionally filtered by model_id and statuses."""
        pass

    @abstractmethod
    def count(self, project_id: Optional[str] = None) -> int:
        """Total number of vector records stored, optionally filtered by project_id."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any underlying storage resources or connections."""
        pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

