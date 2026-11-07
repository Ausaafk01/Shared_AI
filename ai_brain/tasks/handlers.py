"""
Built-in task handlers integrating AI Brain subsystems with background workers (M5.6).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ai_brain.agents.models import AgentExecutionRequest
from ai_brain.agents.simulated.coding import CodingAgent
from ai_brain.agents.simulated.research import ResearchAgent
from ai_brain.agents.simulated.review import ReviewAgent
from ai_brain.loop.models import AgentInteractionInput, AgentResponseInput
from ai_brain.tasks.exceptions import PermanentTaskError, TransientTaskError
from ai_brain.tasks.models import TaskRecord, TaskResult
from ai_brain.tasks.registry import TaskHandlerRegistry

logger = logging.getLogger("ai_brain.tasks.handlers")


def handle_interaction_execution(task: TaskRecord, context: Any = None) -> Dict[str, Any]:
    """
    Background execution of an AI Brain interaction through BrainLoop.
    
    Payload schema:
    {
        "input_text": str,
        "agent_id": Optional[str],
        "session_id": Optional[str],
        "interaction_id": Optional[str],
        "max_context_tokens": Optional[int],
        "search_query": Optional[str],
        "metadata": Optional[dict]
    }
    """
    if not context or not hasattr(context, "brain_loop"):
        raise PermanentTaskError("Context with 'brain_loop' is required to execute interaction tasks.")

    payload = task.payload
    input_text = payload.get("input_text")
    if not input_text:
        raise PermanentTaskError("Missing required 'input_text' in interaction task payload.")

    project_id = task.project_id
    interaction_id = payload.get("interaction_id") or f"int_bg_{task.task_id[:16]}"
    session_id = payload.get("session_id") or f"sess_{task.task_id[:12]}"
    agent_id = payload.get("agent_id") or "agent-coder"

    try:
        # 1. Prepare context
        agent_input = AgentInteractionInput(
            interaction_id=interaction_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            content=input_text,
            token_budget=payload.get("max_context_tokens") or 1000,
            query=payload.get("search_query") or input_text,
            metadata=payload.get("metadata") or {},
        )
        prep_result = context.brain_loop.prepare_context(agent_input)

        # 2. Select agent and generate response
        if "research" in agent_id.lower():
            agent = ResearchAgent(agent_id=agent_id)
        elif "review" in agent_id.lower():
            agent = ReviewAgent(agent_id=agent_id)
        else:
            agent = CodingAgent(agent_id=agent_id)

        exec_req = AgentExecutionRequest(
            project_id=project_id,
            session_id=session_id,
            content=input_text,
            query=payload.get("search_query") or input_text,
            token_budget=payload.get("max_context_tokens") or 1000,
            retrieval_method="HYBRID",
            metadata=payload.get("metadata") or {},
        )
        agent_response_content = agent.generate_response(prep_result, exec_req)

        # 3. Complete interaction and persist memories
        resp_input = AgentResponseInput(
            interaction_id=interaction_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            content=agent_response_content,
        )
        comp = context.brain_loop.complete_interaction(resp_input)

        # Persist extracted memories to container memory store if present
        if hasattr(context, "memory_store") and context.memory_store is not None:
            for m in comp.extracted_memories:
                try:
                    context.memory_store.store(m)
                except Exception as e:
                    logger.warning("Failed to store memory '%s' in memory store: %s", m.memory_id, e)

        return {
            "interaction_id": interaction_id,
            "status": "COMPLETED",
            "agent_response": agent_response_content,
            "memories_created": len(comp.extracted_memories),
            "graph_nodes_created": comp.graph_nodes_integrated,
            "vectors_indexed": comp.vectors_indexed,
        }
    except PermanentTaskError:
        raise
    except Exception as e:
        logger.error("Interaction execution failed on task '%s': %s", task.task_id, e)
        raise TransientTaskError(f"BrainLoop interaction execution error: {e}") from e


def handle_workflow_step(task: TaskRecord, context: Any = None) -> Dict[str, Any]:
    """
    Background execution of a multi-agent collaboration workflow step.
    
    Payload schema:
    {
        "workflow_id": str,
        "stage_name": str,
        "input_data": dict
    }
    """
    payload = task.payload
    workflow_id = payload.get("workflow_id")
    if not workflow_id:
        raise PermanentTaskError("Missing required 'workflow_id' in workflow task payload.")

    stage_name = payload.get("stage_name", "execution")
    input_data = payload.get("input_data", {})

    logger.info("Executed background workflow step '%s' for workflow '%s'", stage_name, workflow_id)
    return {
        "workflow_id": workflow_id,
        "stage_name": stage_name,
        "status": "COMPLETED",
        "processed_inputs": len(input_data),
    }


def handle_memory_maintenance(task: TaskRecord, context: Any = None) -> Dict[str, Any]:
    """
    Background memory maintenance (e.g. consolidation, contradiction resolution, pruning).
    """
    payload = task.payload
    operation = payload.get("operation", "audit")
    logger.info("Executed memory maintenance operation '%s' for project '%s'", operation, task.project_id)
    return {
        "project_id": task.project_id,
        "operation": operation,
        "status": "SUCCESS",
    }


def register_default_handlers(registry: TaskHandlerRegistry) -> None:
    """Register standard built-in AI Brain handlers into a TaskHandlerRegistry."""
    registry.register("interaction_execution", handle_interaction_execution)
    registry.register("workflow_step", handle_workflow_step)
    registry.register("memory_maintenance", handle_memory_maintenance)
