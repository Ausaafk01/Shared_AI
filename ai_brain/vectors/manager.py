"""
VectorMemoryManager orchestrating memory lifecycle integration, version tracking, and search.
"""

import logging
from typing import List, Optional, Sequence, Set, Union

from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus
from ai_brain.vectors.exceptions import ProjectIsolationError, VectorValidationError
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.models import VectorMatch, VectorRecord, VectorSearchResult
from ai_brain.vectors.providers.base import EmbeddingProvider
from ai_brain.vectors.storage.base import VectorStore

logger = logging.getLogger(__name__)


class VectorMemoryManager:
    """
    Coordinates vector embedding generation, storage, memory version updates, and semantic search.
    
    Guarantees:
    - Never mutates the original Memory objects.
    - Preserves project isolation at both indexing and retrieval time.
    - Idempotent: repeated calls do not produce duplicate vectors.
    - Version-aware: marks older versions as SUPERSEDED while preserving historical vectors.
    - Excludes SUPERSEDED and ARCHIVED memories from default active searches.
    - Preserves DISPUTED memory status without treating disputed memories as established facts.
    """

    def __init__(
        self,
        storage: VectorStore,
        provider: EmbeddingProvider,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.index = VectorIndex(storage=self.storage, provider=self.provider)

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def index_memory(self, memory: Memory) -> VectorRecord:
        """
        Embed and index a single Memory object.
        
        Args:
            memory: The immutable Memory instance.
            
        Returns:
            The created or existing VectorRecord.
        """
        if not isinstance(memory, Memory):
            raise VectorValidationError(f"Expected Memory instance, got {type(memory).__name__}")

        project_id = memory.project_id
        memory_id = memory.memory_id
        version = memory.version
        model_id = self.model_id

        # 1. Check if vector record already exists for this exact version and model (idempotency)
        existing = self.storage.get_vector_by_memory(project_id, memory_id, version, model_id)
        if existing:
            # If memory status changed (e.g. became SUPERSEDED or DISPUTED), update record status
            if existing.status != memory.status:
                updated = existing.with_updates(status=memory.status)
                self.storage.update_vector(updated)
                return updated
            return existing

        # 2. Generate embedding vector
        embedding = self.provider.embed_text(memory.content)

        # 3. Construct metadata
        metadata = {
            "memory_type": memory.memory_type,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "root_memory_id": memory.root_memory_id,
            "parent_memory_id": memory.parent_memory_id,
            "source_event_ids": list(memory.source_event_ids),
            "source_agent_ids": list(memory.source_agent_ids),
            "created_at": memory.created_at.isoformat(),
        }
        if memory.status == MemoryStatus.DISPUTED.value:
            metadata["is_disputed"] = True

        # 4. Create and store vector record
        record_id = f"vec:{project_id}:{memory_id}:v{version}:{model_id}"
        record = VectorRecord.create(
            record_id=record_id,
            memory_id=memory_id,
            project_id=project_id,
            version=version,
            model_id=model_id,
            embedding=embedding,
            content=memory.content,
            status=memory.status,
            metadata=metadata,
            created_at=memory.created_at,
        )
        stored_record = self.storage.add_vector(record, fail_if_exists=False)

        # 5. Handle version supersession: if memory has parent_memory_id, supersede parent vector
        if memory.parent_memory_id:
            parent_vec = self.storage.get_vector_by_memory(
                project_id,
                memory.parent_memory_id,
                version=version - 1 if version > 1 else 1,
                model_id=model_id,
            )
            if parent_vec and parent_vec.status != MemoryStatus.SUPERSEDED.value:
                superseded_parent = parent_vec.with_updates(
                    status=MemoryStatus.SUPERSEDED.value,
                    metadata={"superseded_by": memory_id, "superseded_version": version},
                )
                self.storage.update_vector(superseded_parent)

        return stored_record

    def index_memories(self, memories: Sequence[Memory]) -> List[VectorRecord]:
        """
        Embed and index a collection of Memory objects.
        """
        return [self.index_memory(m) for m in memories]

    def archive_memory(self, memory_id: str, project_id: str) -> int:
        """
        Update all vectors associated with a memory_id to ARCHIVED status.
        """
        records = self.storage.get_vectors_by_project(project_id)
        count = 0
        for rec in records:
            if rec.memory_id == memory_id and rec.status != MemoryStatus.ARCHIVED.value:
                archived = rec.with_updates(status=MemoryStatus.ARCHIVED.value)
                self.storage.update_vector(archived)
                count += 1
        return count

    def delete_memory_vectors(self, memory_id: str, project_id: str) -> int:
        """
        Remove all vector records associated with a memory_id from storage.
        """
        return self.storage.delete_vectors_by_memory(project_id, memory_id)

    def search(
        self,
        query: Union[str, Sequence[float]],
        *,
        project_id: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        include_disputed: bool = False,
        allowed_statuses: Optional[Sequence[str]] = None,
    ) -> VectorSearchResult:
        """
        Search for semantically relevant memories in the specified project.
        
        Args:
            query: Query text or precomputed vector.
            project_id: Project identifier enforcing project isolation.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum cosine similarity required.
            include_disputed: If True, includes DISPUTED memories alongside ACTIVE.
            allowed_statuses: Specific statuses to include (overrides include_disputed).
                              Defaults to ['ACTIVE'] (or ['ACTIVE', 'DISPUTED'] if include_disputed=True).
        """
        if allowed_statuses is not None:
            statuses = allowed_statuses
        elif include_disputed:
            statuses = [MemoryStatus.ACTIVE.value, MemoryStatus.DISPUTED.value]
        else:
            statuses = [MemoryStatus.ACTIVE.value]

        return self.index.search(
            query,
            project_id=project_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            allowed_statuses=statuses,
        )

