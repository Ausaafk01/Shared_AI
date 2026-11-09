"""Session continuity & agent coordination router for AI Brain (M5.5).

Provides:
- Creation and retrieval of multi-agent sessions
- Project-scoped session listings
- Agent switching and session handover
- Session closure and audit trail
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ai_brain.agents.sessions.models import SessionRecord, SessionStatus
from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.schemas import (
    ErrorResponse,
    SessionCreateRequest,
    SessionRecordDTO,
    SwitchAgentRequest,
)

logger = logging.getLogger("ai_brain.server.sessions")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/sessions",
    tags=["Sessions"],
)


def _session_to_dto(s: SessionRecord) -> SessionRecordDTO:
    """Map domain SessionRecord to SessionRecordDTO."""
    return SessionRecordDTO(
        session_id=s.session_id,
        project_id=s.project_id,
        primary_agent_id=s.current_agent_id,
        status=s.status.value if hasattr(s.status, "value") else str(s.status),
        summary=s.title,
        metadata=dict(s.metadata or {}),
        created_at=s.created_at.isoformat() if s.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=s.updated_at.isoformat() if s.updated_at else datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "",
    response_model=SessionRecordDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent session",
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def create_session(
    project_id: str,
    payload: SessionCreateRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.SESSION_WRITE)),
    container: BrainServiceContainer = Depends(get_container),
) -> SessionRecordDTO:
    """Create a new session within authorized project scope."""
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    agent_id = payload.primary_agent_id or identity.identity_id

    try:
        session = container.session_manager.create_session(
            project_id=project_id,
            session_id=sid,
            initial_agent_id=agent_id,
            title=payload.summary,
            metadata=payload.metadata,
        )
    except Exception as e:
        logger.error("Failed to create session: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session creation failed: {e}",
        )

    return _session_to_dto(session)


@router.get(
    "/{session_id}",
    response_model=SessionRecordDTO,
    summary="Retrieve session state by identifier",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session(
    project_id: str,
    session_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.SESSION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> SessionRecordDTO:
    """Retrieve session record, validating tenant project isolation."""
    session = container.session_manager.get_session(session_id)
    if not session or session.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found in project '{project_id}'.",
        )

    return _session_to_dto(session)


@router.get(
    "",
    response_model=List[SessionRecordDTO],
    summary="List sessions in project",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def list_sessions(
    project_id: str,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    agent_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.SESSION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> List[SessionRecordDTO]:
    """List sessions matching status and agent filters."""
    stat = SessionStatus(status_filter.upper()) if status_filter and status_filter.upper() in SessionStatus.__members__ else None
    sessions = container.session_manager.list_sessions(
        project_id=project_id,
        status=stat,
        agent_id=agent_id,
        limit=limit,
    )

    return [_session_to_dto(s) for s in sessions]


@router.post(
    "/{session_id}/switch-agent",
    response_model=SessionRecordDTO,
    summary="Handover session to another agent",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid handoff state"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def switch_agent(
    project_id: str,
    session_id: str,
    payload: SwitchAgentRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.SESSION_WRITE)),
    container: BrainServiceContainer = Depends(get_container),
) -> SessionRecordDTO:
    """Handover session to a different agent with recorded rationale."""
    session = container.session_manager.get_session(session_id)
    if not session or session.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found in project '{project_id}'.",
        )

    try:
        updated = container.session_manager.switch_agent(
            session_id=session_id,
            new_agent_id=payload.new_agent_id,
            reason=payload.reason or "API requested handover",
        )
    except Exception as e:
        logger.error("Agent switch failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to switch agent: {e}",
        )

    return _session_to_dto(updated)


@router.post(
    "/{session_id}/close",
    response_model=SessionRecordDTO,
    summary="Close an active session",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def close_session(
    project_id: str,
    session_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.SESSION_WRITE)),
    container: BrainServiceContainer = Depends(get_container),
) -> SessionRecordDTO:
    """Close an active session and mark it completed."""
    session = container.session_manager.get_session(session_id)
    if not session or session.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found in project '{project_id}'.",
        )

    try:
        closed = container.session_manager.close_session(session_id)
    except Exception as e:
        logger.error("Session close failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to close session: {e}",
        )

    return _session_to_dto(closed)
