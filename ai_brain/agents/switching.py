"""
Agent switching coordinator facilitating handover between agents in a project.
"""

from datetime import datetime
import logging
from typing import Optional

from ai_brain.agents.base import Agent
from ai_brain.agents.models import AgentExecutionRequest
from ai_brain.agents.sessions.manager import SessionManager
from ai_brain.events import EventStore, create_event
from ai_brain.events.types import EventType
from ai_brain.loop.models import InteractionContextResult
from ai_brain.loop.orchestrator import BrainLoop

logger = logging.getLogger(__name__)


class AgentSwitcher:
    """
    Coordinates agent transitions within an active project.
    
    Guarantees:
    - Preserves project-scoped context and previous agent's completed work.
    - Prepares relevant shared memories for the incoming agent.
    - Does not expose unrelated private session data across tenants.
    - Persists an immutable AGENT_SWITCH event linking both agents.
    - Works both within a single session and across separate sessions.
    """

    def __init__(
        self,
        *,
        brain_loop: BrainLoop,
        session_manager: Optional[SessionManager] = None,
        event_store: Optional[EventStore] = None,
    ) -> None:
        self.brain_loop = brain_loop
        self.session_manager = session_manager
        self.event_store = event_store or brain_loop.event_store

    def switch(
        self,
        *,
        from_agent_id: str,
        to_agent: Agent,
        project_id: str,
        session_id: str,
        handover_query: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> InteractionContextResult:
        """
        Execute an agent transition, record the AGENT_SWITCH audit event,
        and prepare shared Brain context for the incoming agent.
        """
        # 1. Update session if manager is present
        if self.session_manager is not None:
            self.session_manager.switch_agent(
                session_id=session_id,
                new_agent_id=to_agent.agent_id,
                reason=reason or f"Handover from {from_agent_id} to {to_agent.agent_id}",
            )
        elif self.event_store is not None:
            # Record direct AGENT_SWITCH event
            switch_event = create_event(
                event_type=EventType.AGENT_SWITCH,
                project_id=project_id,
                agent_id=to_agent.agent_id,
                session_id=session_id,
                actor=f"agent:{from_agent_id}",
                metadata={
                    "from_agent": from_agent_id,
                    "to_agent": to_agent.agent_id,
                    "reason": reason or "Agent handover",
                },
            )
            self.event_store.store_event(switch_event)

        # 2. Prepare context for incoming agent based on shared project memories
        query = handover_query or f"Previous findings and decisions from {from_agent_id}"
        req = AgentExecutionRequest(
            project_id=project_id,
            session_id=session_id,
            content=f"Handover to {to_agent.name} ({to_agent.agent_type.value}): {reason or 'Continue task'}",
            query=query,
        )
        context_result = to_agent.prepare_context(req, self.brain_loop)

        logger.info(
            "Agent switched on project '%s' (session '%s'): %s -> %s",
            project_id,
            session_id,
            from_agent_id,
            to_agent.agent_id,
        )
        return context_result

