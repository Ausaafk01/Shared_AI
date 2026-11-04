"""
Collaboration Workflow Engine executing ordered multi-agent task lifecycles in AI Brain.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Mapping, Optional
import uuid

from ai_brain.agents.base import Agent
from ai_brain.agents.collaboration.models import (
    CollaborationStep,
    CollaborationStepResult,
    CollaborationWorkflowRecord,
    WorkflowStatus,
)
from ai_brain.agents.exceptions import AgentNotFoundError, CollaborationWorkflowError
from ai_brain.agents.models import AgentExecutionRequest
from ai_brain.loop.orchestrator import BrainLoop

logger = logging.getLogger(__name__)


class CollaborationEngine:
    """
    Coordinates multi-agent collaboration workflows.
    
    Guarantees:
    - Ordered execution of heterogeneous agents on a shared project task.
    - Seamless context passing between agents via AI Brain's shared memory.
    - Durable tracking of step progress and outputs.
    - Resumption of interrupted or failed workflows without re-executing completed steps.
    - Preservation of conflicting perspectives through Brain Conflict Management.
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, CollaborationWorkflowRecord] = {}

    def get_workflow(self, workflow_id: str) -> Optional[CollaborationWorkflowRecord]:
        """Look up a workflow record by ID."""
        return self._workflows.get(workflow_id)

    def create_workflow(
        self,
        *,
        project_id: str,
        session_id: str,
        task_description: str,
        steps: List[CollaborationStep],
        workflow_id: Optional[str] = None,
    ) -> CollaborationWorkflowRecord:
        """Create and register a new collaboration workflow."""
        wid = workflow_id.strip() if workflow_id else str(uuid.uuid4())
        record = CollaborationWorkflowRecord(
            workflow_id=wid,
            project_id=project_id,
            session_id=session_id,
            task_description=task_description,
            steps=tuple(steps),
            completed_steps=(),
            status=WorkflowStatus.PENDING,
        )
        self._workflows[wid] = record
        return record

    def create_research_coding_review_workflow(
        self,
        *,
        project_id: str,
        session_id: str,
        task_description: str,
        research_agent_id: str = "agent-researcher",
        coding_agent_id: str = "agent-coder",
        review_agent_id: str = "agent-reviewer",
    ) -> CollaborationWorkflowRecord:
        """
        Convenience factory creating a standard 3-stage Research -> Coding -> Review workflow.
        """
        steps = [
            CollaborationStep(
                step_index=1,
                name="Research & Fact Gathering",
                agent_id=research_agent_id,
                instruction=f"Investigate architectural facts and requirements for: {task_description}",
                query=task_description,
            ),
            CollaborationStep(
                step_index=2,
                name="Technical Implementation & Decision",
                agent_id=coding_agent_id,
                instruction=f"Formulate implementation decisions and task plans based on research for: {task_description}",
                query=f"{task_description} database architecture implementation",
            ),
            CollaborationStep(
                step_index=3,
                name="Review & Contradiction Evaluation",
                agent_id=review_agent_id,
                instruction=f"Review the implementation decisions against established project context for: {task_description}",
                query=f"{task_description} decisions and facts",
            ),
        ]
        return self.create_workflow(
            project_id=project_id,
            session_id=session_id,
            task_description=task_description,
            steps=steps,
        )

    def execute_workflow(
        self,
        workflow: CollaborationWorkflowRecord,
        agents: Mapping[str, Agent],
        brain_loop: BrainLoop,
    ) -> CollaborationWorkflowRecord:
        """
        Execute an ordered multi-agent collaboration workflow.
        """
        wid = workflow.workflow_id
        results_by_index = {r.step_index: r for r in workflow.completed_steps}
        completed_indices = {r.step_index for r in workflow.completed_steps if r.status == "COMPLETED"}

        def get_current_results() -> tuple[CollaborationStepResult, ...]:
            return tuple(results_by_index[k] for k in sorted(results_by_index.keys()))

        current_record = CollaborationWorkflowRecord(
            workflow_id=wid,
            project_id=workflow.project_id,
            session_id=workflow.session_id,
            task_description=workflow.task_description,
            steps=workflow.steps,
            completed_steps=get_current_results(),
            status=WorkflowStatus.IN_PROGRESS,
            updated_at=datetime.now(timezone.utc),
        )
        self._workflows[wid] = current_record

        for step in workflow.steps:
            if step.step_index in completed_indices:
                logger.info("Skipping already completed step %d ('%s')", step.step_index, step.name)
                continue

            agent = agents.get(step.agent_id)
            if agent is None:
                err_msg = f"Agent '{step.agent_id}' for step {step.step_index} not found in available agents."
                step_res = CollaborationStepResult(
                    step_index=step.step_index,
                    name=step.name,
                    agent_id=step.agent_id,
                    status="FAILED",
                    error_message=err_msg,
                )
                results_by_index[step.step_index] = step_res
                failed_record = CollaborationWorkflowRecord(
                    workflow_id=wid,
                    project_id=workflow.project_id,
                    session_id=workflow.session_id,
                    task_description=workflow.task_description,
                    steps=workflow.steps,
                    completed_steps=get_current_results(),
                    status=WorkflowStatus.FAILED,
                    error_message=err_msg,
                    updated_at=datetime.now(timezone.utc),
                )
                self._workflows[wid] = failed_record
                raise AgentNotFoundError(err_msg)

            # Build execution request for the agent
            req = AgentExecutionRequest(
                project_id=workflow.project_id,
                session_id=workflow.session_id,
                content=step.instruction,
                query=step.query,
                metadata={"step_index": step.step_index, "step_name": step.name, "workflow_id": wid},
            )

            try:
                exec_resp = agent.execute(req, brain_loop)
                mem_count = len(exec_resp.completion_result.extracted_memories) if exec_resp.completion_result else 0
                conf_count = len(exec_resp.completion_result.conflicts_detected) if exec_resp.completion_result else 0

                step_res = CollaborationStepResult(
                    step_index=step.step_index,
                    name=step.name,
                    agent_id=step.agent_id,
                    status="COMPLETED",
                    output_content=exec_resp.content,
                    interaction_id=exec_resp.interaction_id,
                    memories_created_count=mem_count,
                    conflicts_detected_count=conf_count,
                )
                results_by_index[step.step_index] = step_res
                completed_indices.add(step.step_index)

                # Update in-progress progress
                self._workflows[wid] = CollaborationWorkflowRecord(
                    workflow_id=wid,
                    project_id=workflow.project_id,
                    session_id=workflow.session_id,
                    task_description=workflow.task_description,
                    steps=workflow.steps,
                    completed_steps=get_current_results(),
                    status=WorkflowStatus.IN_PROGRESS,
                    updated_at=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.error("Step %d ('%s') failed: %s", step.step_index, step.name, e)
                step_res = CollaborationStepResult(
                    step_index=step.step_index,
                    name=step.name,
                    agent_id=step.agent_id,
                    status="FAILED",
                    error_message=str(e),
                )
                results_by_index[step.step_index] = step_res
                failed_record = CollaborationWorkflowRecord(
                    workflow_id=wid,
                    project_id=workflow.project_id,
                    session_id=workflow.session_id,
                    task_description=workflow.task_description,
                    steps=workflow.steps,
                    completed_steps=get_current_results(),
                    status=WorkflowStatus.FAILED,
                    error_message=f"Step {step.step_index} failed: {e}",
                    updated_at=datetime.now(timezone.utc),
                )
                self._workflows[wid] = failed_record
                if step.required:
                    raise CollaborationWorkflowError(f"Required workflow step {step.step_index} failed: {e}") from e

        final_record = CollaborationWorkflowRecord(
            workflow_id=wid,
            project_id=workflow.project_id,
            session_id=workflow.session_id,
            task_description=workflow.task_description,
            steps=workflow.steps,
            completed_steps=get_current_results(),
            status=WorkflowStatus.COMPLETED,
            updated_at=datetime.now(timezone.utc),
        )
        self._workflows[wid] = final_record
        return final_record

    def resume_workflow(
        self,
        workflow_id: str,
        agents: Mapping[str, Agent],
        brain_loop: BrainLoop,
    ) -> CollaborationWorkflowRecord:
        """Resume an interrupted or failed workflow from its last uncompleted step."""
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            raise CollaborationWorkflowError(f"Workflow '{workflow_id}' not found for resumption.")

        return self.execute_workflow(workflow, agents, brain_loop)
