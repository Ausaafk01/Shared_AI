"""Background task submission and status retrieval endpoints (M5.7).

Provides external HTTP clients and agents with versioned REST endpoints
to submit asynchronous durable tasks and inspect their execution state.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.schemas import (
    ErrorResponse,
    TaskStatusResponse,
    TaskSubmitRequest,
    TaskSubmitResponse,
)
from ai_brain.tasks.exceptions import DuplicateTaskError

logger = logging.getLogger("ai_brain.server.routers.tasks")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/tasks",
    tags=["Background Tasks"],
)


@router.post(
    "",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous background task to the queue",
    responses={
        400: {"model": ErrorResponse, "description": "Validation or bad request error"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authentication"},
        403: {"model": ErrorResponse, "description": "Permission denied for project"},
        409: {"model": ErrorResponse, "description": "Duplicate task idempotency conflict"},
    },
)
async def submit_task(
    project_id: str,
    payload: TaskSubmitRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_EXECUTE)),
    container: BrainServiceContainer = Depends(get_container),
) -> TaskSubmitResponse:
    """Submits a durable background task to the persistent PostgreSQL/SQLite queue."""
    if not container.task_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background task subsystem is not configured.",
        )

    try:
        task_data = dict(payload.payload)
        if payload.idempotency_key and "idempotency_key" not in task_data:
            task_data["idempotency_key"] = payload.idempotency_key

        task = container.task_store.submit_task(
            project_id=project_id,
            task_type=payload.task_type,
            payload=task_data,
            idempotency_key=payload.idempotency_key,
        )
        return TaskSubmitResponse(
            task_id=task.task_id,
            project_id=task.project_id,
            task_type=task.task_type,
            status=task.status.value,
            idempotency_key=task.idempotency_key,
            created_at=task.created_at.isoformat(),
        )
    except DuplicateTaskError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate task submission: {e}",
        )
    except Exception as e:
        logger.error("Failed to submit task for project '%s': %s", project_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task submission failed: {e}",
        )


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve the status and execution result of a background task",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid authentication"},
        403: {"model": ErrorResponse, "description": "Permission denied for project"},
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
)
async def get_task_status(
    project_id: str,
    task_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> TaskStatusResponse:
    """Retrieves current processing state, retry counter, and result data for a task."""
    if not container.task_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background task subsystem is not configured.",
        )

    task = container.task_store.get_task(project_id, task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found in project '{project_id}'.",
        )

    return TaskStatusResponse(
        task_id=task.task_id,
        project_id=task.project_id,
        task_type=task.task_type,
        status=task.status.value,
        retry_count=task.retry_count,
        idempotency_key=task.idempotency_key,
        result_data=task.payload.get("result"),
        error_message=task.error_message,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )
