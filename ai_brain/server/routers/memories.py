"""Memory management router for AI Brain (M5.5).

Provides:
- Memory creation and retrieval
- Project-scoped memory listing with status and type filters
- Memory update with versioning chains, provenance consolidation, and deduplication
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.schemas import (
    ErrorResponse,
    MemoryCreateRequest,
    MemoryDTO,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
)

logger = logging.getLogger("ai_brain.server.memories")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/memories",
    tags=["Memories"],
)


def _memory_to_dto(m: Memory) -> MemoryDTO:
    """Format domain Memory object into MemoryDTO."""
    return MemoryDTO(
        memory_id=m.memory_id,
        project_id=m.project_id,
        content=m.content,
        memory_type=m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
        status=m.status.value if hasattr(m.status, "value") else str(m.status),
        version=m.version,
        root_memory_id=m.root_memory_id,
        parent_memory_id=m.parent_memory_id,
        importance=m.importance,
        importance_score=m.importance,
        confidence=m.confidence,
        source_event_ids=list(m.source_event_ids or []),
        source_agent_ids=list(m.source_agent_ids or []),
        metadata=dict(m.metadata or {}),
        created_at=m.created_at.isoformat() if m.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=m.updated_at.isoformat() if m.updated_at else datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "",
    response_model=MemoryDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new persistent knowledge memory",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid memory payload"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def create_memory(
    project_id: str,
    payload: MemoryCreateRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.MEMORY_WRITE)),
    container: BrainServiceContainer = Depends(get_container),
) -> MemoryDTO:
    """Create a new immutable knowledge memory, automatically updating vector and graph stores."""
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    agent_id = payload.source_agent_id or identity.identity_id

    # Parse memory type
    m_type_str = payload.memory_type.upper().strip()
    try:
        mem_type = MemoryType(m_type_str)
    except ValueError:
        mem_type = MemoryType.FACT

    memory = Memory.create(
        memory_id=mem_id,
        project_id=project_id,
        content=payload.content,
        memory_type=mem_type,
        importance=payload.importance if payload.importance is not None else 0.5,
        source_agent_ids=[agent_id] if agent_id else [],
        metadata=payload.metadata or {},
    )

    # 1. Store memory
    container.memory_store.store(memory)

    # 2. Update vector store if vector manager configured
    if container.vector_manager:
        try:
            container.vector_manager.index_memory(memory)
        except Exception as e:
            logger.warning("Failed to index memory '%s' in vector store: %s", mem_id, e)

    # 3. Update graph if graph configured
    if container.graph:
        try:
            from ai_brain.graph.models import GraphNode
            from ai_brain.graph.types import NodeType
            node = GraphNode(
                node_id=mem_id,
                project_id=project_id,
                node_type=NodeType.MEMORY.value,
                label=memory.content[:50],
                metadata={"memory_id": mem_id, "importance": memory.importance, "memory_type": mem_type.value},
            )
            container.graph.add_node(node)
        except Exception as e:
            logger.warning("Failed to integrate memory '%s' in graph: %s", mem_id, e)

    return _memory_to_dto(memory)


@router.get(
    "/{memory_id}",
    response_model=MemoryDTO,
    summary="Retrieve a memory by identifier",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
    },
)
async def get_memory(
    project_id: str,
    memory_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.MEMORY_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> MemoryDTO:
    """Retrieve memory version by identifier, ensuring strict project boundary."""
    memory = container.memory_store.get(memory_id)
    if not memory or memory.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found in project '{project_id}'.",
        )

    return _memory_to_dto(memory)


@router.get(
    "",
    response_model=List[MemoryDTO],
    summary="List project memories with filtering",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def list_memories(
    project_id: str,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    memory_type_filter: Optional[str] = Query(default=None, alias="memory_type"),
    include_historical: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.MEMORY_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> List[MemoryDTO]:
    """List memories for project with status, type, and historical inclusion options."""
    memories = container.memory_store.list(project_id=project_id, limit=limit)

    # Filter status
    if status_filter:
        memories = [m for m in memories if m.status.upper() == status_filter.upper()]
    elif not include_historical:
        # Default excludes SUPERSEDED and ARCHIVED
        memories = [m for m in memories if m.status == MemoryStatus.ACTIVE.value or m.status == MemoryStatus.DISPUTED.value]

    # Filter type
    if memory_type_filter:
        memories = [m for m in memories if m.memory_type.upper() == memory_type_filter.upper()]

    return [_memory_to_dto(m) for m in memories[:limit]]


@router.post(
    "/{memory_id}/update",
    response_model=MemoryUpdateResponse,
    summary="Update a memory creating a new version in the chain",
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
    },
)
async def update_memory(
    project_id: str,
    memory_id: str,
    payload: MemoryUpdateRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.MEMORY_UPDATE)),
    container: BrainServiceContainer = Depends(get_container),
) -> MemoryUpdateResponse:
    """Updates memory content following immutable version chain rules (Rule A-D)."""
    existing = container.memory_store.get(memory_id)
    if not existing or existing.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found in project '{project_id}'.",
        )

    # Construct candidate incoming memory
    incoming = Memory.create(
        memory_id=f"mem_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        content=payload.content,
        memory_type=existing.memory_type,
        importance=payload.importance if payload.importance is not None else existing.importance,
        source_event_ids=payload.source_event_ids or existing.source_event_ids,
        source_agent_ids=payload.source_agent_ids or [identity.identity_id],
        metadata={**dict(existing.metadata or {}), **(payload.metadata or {})},
    )

    # Process through MemoryUpdater
    result = container.memory_updater.update(existing, incoming)

    # Persist affected memories (e.g. superseded old memory, active new memory)
    for aff in result.affected_memories:
        container.memory_store.store(aff)

    target_memory = result.current_memory or existing

    # Update vector index if superseded or created
    if container.vector_manager and result.created and result.current_memory:
        try:
            container.vector_manager.index_memory(result.current_memory)
        except Exception as e:
            logger.warning("Failed to index updated memory in vector store: %s", e)

    return MemoryUpdateResponse(
        result_type=result.action.value if hasattr(result.action, "value") else str(result.action),
        memory=_memory_to_dto(target_memory),
        version=target_memory.version,
        superseded_memory_id=existing.memory_id if result.superseded else None,
    )
