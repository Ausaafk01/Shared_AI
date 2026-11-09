"""Multi-agent collaboration workflow router for AI Brain (M5.5).

Exposes:
- Standard multi-agent workflows (Research -> Coding -> Review)
- Custom agent collaboration sequences
- Step execution, agent handoffs, failure handling, and resumption
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from ai_brain.agents.collaboration.models import (
    CollaborationStep,
    CollaborationWorkflowRecord,
)
from ai_brain.agents.simulated import CodingAgent, ResearchAgent, ReviewAgent
from ai_brain.auth import AuthenticatedIdentity, Permission
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    get_container,
    require_permission,
)
from ai_brain.server.schemas import (
    CollaborationStepResultDTO,
    ErrorResponse,
    WorkflowCreateRequest,
    WorkflowExecuteRequest,
    WorkflowExecutionResponse,
    WorkflowRecordDTO,
)

logger = logging.getLogger("ai_brain.server.workflows")

router = APIRouter(
    prefix="/api/v1/projects/{project_id}/workflows",
    tags=["Workflows"],
)


def _workflow_to_dto(w: CollaborationWorkflowRecord) -> WorkflowRecordDTO:
    """Format CollaborationWorkflowRecord to WorkflowRecordDTO."""
    return WorkflowRecordDTO(
        workflow_id=w.workflow_id,
        project_id=w.project_id,
        session_id=w.session_id,
        task_description=w.task_description,
        status=w.status.value if hasattr(w.status, "value") else str(w.status),
        step_count=len(w.steps),
        completed_step_count=len(w.completed_steps),
        created_at=w.created_at.isoformat() if w.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=w.updated_at.isoformat() if w.updated_at else datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "",
    response_model=WorkflowRecordDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create a multi-agent collaboration workflow",
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
    },
)
async def create_workflow(
    project_id: str,
    payload: WorkflowCreateRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_EXECUTE)),
    container: BrainServiceContainer = Depends(get_container),
) -> WorkflowRecordDTO:
    """Instantiate a multi-agent workflow definition."""
    session_id = payload.session_id or f"sess_wf_{uuid.uuid4().hex[:8]}"

    if payload.workflow_type.upper() == "RESEARCH_CODING_REVIEW":
        wf = container.collaboration_engine.create_research_coding_review_workflow(
            project_id=project_id,
            session_id=session_id,
            task_description=payload.task_description,
        )
    else:
        # Custom workflow steps
        steps = []
        if payload.steps:
            for s in payload.steps:
                steps.append(
                    CollaborationStep(
                        step_index=s.step_index,
                        name=s.name,
                        agent_id=s.agent_id,
                        instruction=s.instruction,
                        query=s.query or payload.task_description,
                    )
                )
        wf = container.collaboration_engine.create_workflow(
            project_id=project_id,
            session_id=session_id,
            task_description=payload.task_description,
            steps=steps,
        )

    return _workflow_to_dto(wf)


@router.get(
    "/{workflow_id}",
    response_model=WorkflowRecordDTO,
    summary="Retrieve workflow state",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_workflow(
    project_id: str,
    workflow_id: str,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_READ)),
    container: BrainServiceContainer = Depends(get_container),
) -> WorkflowRecordDTO:
    """Retrieve workflow record, verifying tenant scope."""
    wf = container.collaboration_engine.get_workflow(workflow_id)
    if not wf or wf.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found in project '{project_id}'.",
        )

    return _workflow_to_dto(wf)


@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowExecutionResponse,
    summary="Execute multi-agent workflow steps with agent handoffs",
    responses={
        400: {"model": ErrorResponse, "description": "Execution failure"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def execute_workflow(
    project_id: str,
    workflow_id: str,
    payload: WorkflowExecuteRequest,
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.INTERACTION_EXECUTE)),
    container: BrainServiceContainer = Depends(get_container),
) -> WorkflowExecutionResponse:
    """Execute workflow steps across collaborating agents."""
    wf = container.collaboration_engine.get_workflow(workflow_id)
    if not wf or wf.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found in project '{project_id}'.",
        )

    # Initialize simulated collaborating agents
    agents_map = {
        "agent-researcher": ResearchAgent("agent-researcher", "Research Specialist"),
        "agent-coder": CodingAgent("agent-coder", "Coding Specialist"),
        "agent-reviewer": ReviewAgent("agent-reviewer", "Review Specialist"),
    }

    try:
        completed_wf = container.collaboration_engine.execute_workflow(
            workflow=wf,
            agents=agents_map,
            brain_loop=container.brain_loop,
        )
    except Exception as e:
        logger.error("Workflow execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {e}",
        )

    step_results = [
        CollaborationStepResultDTO(
            step_index=r.step_index,
            name=r.name,
            agent_id=r.agent_id,
            status=r.status,
            output=r.output_content,
            error_message=r.error_message,
        )
        for r in completed_wf.completed_steps
    ]

    last_output = (
        completed_wf.completed_steps[-1].output_content
        if completed_wf.completed_steps
        else None
    )

    return WorkflowExecutionResponse(
        workflow_id=completed_wf.workflow_id,
        project_id=completed_wf.project_id,
        session_id=completed_wf.session_id,
        status=completed_wf.status.value if hasattr(completed_wf.status, "value") else str(completed_wf.status),
        task_description=completed_wf.task_description,
        completed_steps=step_results,
        final_output=last_output,
    )
