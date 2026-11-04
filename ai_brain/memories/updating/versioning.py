"""
Version history tracking and chain management for Memory instances.
"""

from typing import Dict, List, Optional

from ai_brain.memories.exceptions import (
    InvalidUpdateOperationError,
    VersionConflictError,
)
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus


class MemoryVersionHistory:
    """
    Maintains an immutable version chain for a single conceptual memory entity.
    
    Distinction:
    - Stable Identity (root_memory_id): Invariant identifier representing the conceptual
      knowledge entity across all revisions.
    - Version ID (memory_id): Unique identifier for a specific immutable snapshot at a
      point in time.
    - Version Number (version): Monotonically increasing integer (1, 2, 3, ...) providing
      deterministic linear ordering across versions.
    - Previous Version Pointer (parent_memory_id): Explicit link to immediate ancestor.
    """

    def __init__(self, root_memory_id: str) -> None:
        if not root_memory_id or not isinstance(root_memory_id, str):
            raise InvalidUpdateOperationError("root_memory_id must be a non-empty string")
        self.root_memory_id = root_memory_id.strip()
        self._versions: List[Memory] = []

    @property
    def version_count(self) -> int:
        """Total number of recorded versions."""
        return len(self._versions)

    def add_version(self, memory: Memory) -> None:
        """
        Append a new memory snapshot to the version history.
        
        Args:
            memory: Validated immutable Memory instance.
            
        Raises:
            VersionConflictError: If the memory does not belong to this root or has an invalid version number.
        """
        if memory.root_memory_id != self.root_memory_id:
            raise VersionConflictError(
                f"Memory root_memory_id '{memory.root_memory_id}' does not match history root '{self.root_memory_id}'"
            )

        if self._versions:
            latest = self._versions[-1]
            if memory.version <= latest.version:
                raise VersionConflictError(
                    f"New version {memory.version} must be strictly greater than latest version {latest.version}"
                )
            if memory.parent_memory_id != latest.memory_id:
                raise VersionConflictError(
                    f"New version parent_memory_id '{memory.parent_memory_id}' does not match predecessor ID '{latest.memory_id}'"
                )
        else:
            if memory.version != 1:
                # First recorded version
                pass

        self._versions.append(memory)

    def get_current(self) -> Optional[Memory]:
        """
        Return the current version of this memory.
        
        Prioritizes the active version; falls back to the latest version.
        """
        if not self._versions:
            return None
        # Check from latest backwards for an ACTIVE version
        for v in reversed(self._versions):
            if v.status == MemoryStatus.ACTIVE.value:
                return v
        return self._versions[-1]

    def get_version(self, version_num: int) -> Optional[Memory]:
        """Retrieve a specific version by its version number (1-indexed)."""
        for v in self._versions:
            if v.version == version_num:
                return v
        return None

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific version by its unique memory_id."""
        for v in self._versions:
            if v.memory_id == memory_id:
                return v
        return None

    def get_history(self) -> List[Memory]:
        """Return the complete chronological list of versions."""
        return list(self._versions)

    def is_current(self, memory_id: str) -> bool:
        """Return True if the memory_id corresponds to the current version."""
        current = self.get_current()
        return current is not None and current.memory_id == memory_id


class VersionTracker:
    """
    In-memory registry of version histories for all known memories.
    """

    def __init__(self) -> None:
        self._histories: Dict[str, MemoryVersionHistory] = {}
        self._id_to_root: Dict[str, str] = {}

    def track(self, memory: Memory) -> MemoryVersionHistory:
        """Register a memory version into the appropriate version chain."""
        root_id = memory.root_memory_id or memory.memory_id
        if root_id not in self._histories:
            self._histories[root_id] = MemoryVersionHistory(root_id)

        history = self._histories[root_id]
        if not history.get_by_id(memory.memory_id):
            history.add_version(memory)
            self._id_to_root[memory.memory_id] = root_id
        return history

    def get_history(self, root_memory_id: str) -> Optional[MemoryVersionHistory]:
        """Retrieve the version history for a given root memory ID."""
        return self._histories.get(root_memory_id)

    def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """Find a specific memory version by its individual memory_id."""
        root_id = self._id_to_root.get(memory_id)
        if not root_id:
            return None
        history = self._histories.get(root_id)
        return history.get_by_id(memory_id) if history else None

    def get_current(self, root_memory_id: str) -> Optional[Memory]:
        """Get the current active memory for a given root memory ID."""
        history = self._histories.get(root_memory_id)
        return history.get_current() if history else None

    def all_current(self) -> List[Memory]:
        """Return the current version for all tracked memories."""
        results = []
        for history in self._histories.values():
            curr = history.get_current()
            if curr:
                results.append(curr)
        return results

