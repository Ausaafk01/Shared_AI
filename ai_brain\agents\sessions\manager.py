"""
Session Manager providing session continuity, lifecycle tracking, and agent switching.
"""

from datetime import datetime, timezone
import logging
from typing import Any, List, Mapping, Optional
import uuid

from ai_brain.agents.exceptions import SessionNotFoundError, SessionStateError
from ai_brain.agents.sessions.models import SessionRecord, SessionStatus
from ai_brain.agents.sessions.storage.base import SessionStore
from ai_brain.agents.sessions.storage.sqlite import SQLiteSessionStore
from ai_brain.events import EventStore, create_event
from ai_brain.events.types import EventType
from ai_brain.loop.models import InteractionRecord
from ai_brain.loop.orchestrator import BrainLoop

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session lifecycle, continuity across restarts, and agent switching.
    
    Guarantees:
    - Session records survive process restarts in persistent storage.
    - Emits immutable SESSION_START, AGENT_SWITCH, and SESSION_END audit events.
    - Clear boundaries: project-level knowledge vs agent identity vs session history.
    - Preserves previous agent work when switching active agents.
    """

    def __init__(
        self,
        *,
        store: Optional[SessionStore] = None,
        event_store: Optional[EventStore] = None,
    ) -> None:
        self.store = store or SQLiteSessionStore(":memory:")
        self.event_store = event_store

    def create_session(
        self,
        project_id: str,
        *,
        session_id: Optional[str] = None,
        initial_agent_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SessionRecord:
        """
        Create and persist a new session, recording a SESSION_START event.
        """
        sid = session_id.strip() if session_id else str(uuid.uuid4())
        session = SessionRecord(
            session_id=sid,
            project_id=project_id,
            title=title or f"Session {sid[:8]}",
            current_agent_id=initial_agent_id,
            participating_agent_ids=(initial_agent_id,) if initial_agent_id else (),
            status=SessionStatus.ACTIVE,
            metadata=metadata or {},
        )
        self.store.save(session)

        # Emit audit event to EventStore
        if self.event_store is not None:
            try:
                start_event = create_event(
                    event_type=EventType.SESSION_START,
                    project_id=project_id,
                    agent_id=initial_agent_id,
                    session_id=sid,
                    actor=f"agent:{initial_agent_id}" if initial_agent_id else "system",
                    input={"title": session.title, "metadata": dict(session.metadata)},
                )
                self.event_store.store_event(start_event)
            except Exception as e:
                logger.warning("Failed to record SESSION_START event for session '%s': %s", sid, e)

        logger.info("Created session '%s' in project '%s' (initial_agent=%s)", sid, project_id, initial_agent_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session record by its ID."""
        return self.store.get(session_id)

    def get_or_raise(self, session_id: str) -> SessionRecord:
        """Retrieve session or raise SessionNotFoundError."""
        session = self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return session

    def resume_session(self, session_id: str) -> SessionRecord:
        """
        Resume a previous session, transitioning its status to ACTIVE if paused.
        """
        session = self.get_or_raise(session_id)
        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError(f"Cannot resume completed session '{session_id}'.")

        if session.status != SessionStatus.ACTIVE:
            session = session.with_status(SessionStatus.ACTIVE)
            self.store.save(session)

        logger.info("Resumed session '%s' in project '%s'", session_id, session.project_id)
        return session

    def switch_agent(
        self,
        session_id: str,
        new_agent_id: str,
        *,
        reason: Optional[str] = None,
    ) -> SessionRecord:
        """
        Switch the active agent within a session and record an AGENT_SWITCH audit event.
        """
        session = self.get_or_raise(session_id)
        old_agent_id = session.current_agent_id

        if old_agent_id == new_agent_id:
            return session

        updated_session = session.with_agent(new_agent_id)
        self.store.save(updated_session)

        # Record AGENT_SWITCH event in EventStore
        if self.event_store is not None:
            try:
                switch_event = create_event(
                    event_type=EventType.AGENT_SWITCH,
                    project_id=session.project_id,
                    agent_id=new_agent_id,
                    session_id=session.session_id,
                    actor="system:session-manager",
                    metadata={
                        "from_agent": old_agent_id,
                        "to_agent": new_agent_id,
                        "reason": reason or "Agent handover",
                    },
                )
                self.event_store.store_event(switch_event)
            except Exception as e:
                logger.warning("Failed to record AGENT_SWITCH event: %s", e)

        logger.info("Switched agent in session '%s': %s -> %s", session_id, old_agent_id, new_agent_id)
        return updated_session

    def close_session(self, session_id: str) -> SessionRecord:
        """
        Mark a session as COMPLETED and emit a SESSION_END event.
        """
        session = self.get_or_raise(session_id)
        completed_session = session.with_status(SessionStatus.COMPLETED)
        self.store.save(completed_session)

        if self.event_store is not None:
            try:
                end_event = create_event(
                    event_type=EventType.SESSION_END,
                    project_id=session.project_id,
                    agent_id=session.current_agent_id,
                    session_id=session.session_id,
                    actor="system:session-manager",
                    output={"status": "COMPLETED"},
                )
                self.event_store.store_event(end_event)
            except Exception as e:
                logger.warning("Failed to record SESSION_END event: %s", e)

        logger.info("Closed session '%s'", session_id)
        return completed_session

    def list_sessions(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
    ) -> List[SessionRecord]:
        """List and filter session records."""
        return self.store.list(project_id=project_id, agent_id=agent_id, status=status)

    def get_session_history(
        self,
        session_id: str,
        brain_loop: BrainLoop,
    ) -> List[InteractionRecord]:
        """Retrieve interaction records belonging strictly to this session."""
        return brain_loop.list_interactions(session_id=session_id)

    def close(self) -> None:
        """Close underlying storage connections."""
        self.store.close()

