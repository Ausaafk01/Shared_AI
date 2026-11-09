"""Retrieval router for AI Brain (M5.5).

Exposes multi-strategy memory search:
- Hybrid vector + graph retrieval with reciprocal rank fusion (RRF)
- Semantic vector cosine search
- Knowledge graph entity traversal
- Attributed scoring and evidence tracking
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.retrieval.models import QueryFilters, RetrievalMethod
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.schemas import (
    ErrorResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RetrievalResultItemDTO,
)

logger = logging.getLogger("ai_brain.server.retrieval")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/retrieval",
    tags=["Retrieval"],
)


@router.post(
    "/query",
    response_model=RetrievalQueryResponse,
    summary="Execute multi-strategy knowledge retrieval",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def query_retrieval(
    project_id: str,
    payload: RetrievalQueryRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.VECTOR_SEARCH)),
    container: BrainServiceContainer = Depends(get_container),
) -> RetrievalQueryResponse:
    """Retrieves relevant memories using vector, graph, or hybrid search with attribution."""
    # Normalize strategy
    strategy_str = (payload.strategy or "HYBRID").upper().strip()
    try:
        method = RetrievalMethod(strategy_str)
    except ValueError:
        method = RetrievalMethod.HYBRID

    # Build allowed statuses
    allowed_statuses = ["ACTIVE", "DISPUTED"]
    if payload.include_superseded:
        allowed_statuses.append("SUPERSEDED")
    if payload.include_archived:
        allowed_statuses.append("ARCHIVED")

    filters = QueryFilters(
        allowed_statuses=tuple(allowed_statuses),
    )

    try:
        resp = container.retrieval_engine.retrieve(
            query=payload.query_text,
            project_id=project_id,
            method=method,
            top_k=payload.limit,
            similarity_threshold=payload.similarity_threshold,
            filters=filters,
        )
    except Exception as e:
        logger.error("Retrieval failed for project '%s': %s", project_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval query failed: {e}",
        )

    items_dto = []
    for item in resp.results:
        trav_info = dict(item.graph_metadata) if item.graph_metadata else None
        sources = [s.value if hasattr(s, "value") else str(s) for s in item.sources]
        items_dto.append(
            RetrievalResultItemDTO(
                memory_id=item.memory_id,
                content=item.content,
                score=item.score,
                sources=sources,
                memory_type=item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
                status=item.status.value if hasattr(item.status, "value") else str(item.status),
                vector_score=item.vector_score,
                graph_score=item.graph_score,
                graph_traversal_info=trav_info,
            )
        )

    return RetrievalQueryResponse(
        query=resp.query,
        strategy=resp.method,
        items=items_dto,
        warnings=list(resp.warnings),
        count=len(items_dto),
    )
