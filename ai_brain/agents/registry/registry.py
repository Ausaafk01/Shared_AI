"""
Agent Registry managing lifecycle, configuration, and discovery of AI Brain agents.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

from ai_brain.agents.base import Agent
from ai_brain.agents.exceptions import (
    AgentNotFoundError,
    AgentRegistrationError,
    DuplicateAgentError,
)
from ai_brain.agents.models import AgentRecord, AgentStatus, AgentType, _validate_no_secrets
from ai_brain.agents.registry.base import AgentRegistryStore
from ai_brain.agents.registry.sqlite import SQLiteAgentRegistryStore

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry responsible for:
    - Registering agents (preventing duplicate agent registration)
    - Retrieving agents by ID
    - Listing available agents with filtering
    - Activating, deactivating, and updating agent statuses
    - Updating agent configuration (forbidding API keys or secrets)
    - Ensuring agent identities persist across application restarts
    """

    def __init__(self, store: Optional[AgentRegistryStore] = None) -> None:
        self.store = store or SQLiteAgentRegistryStore(":memory:")
        self._instances: Dict[str, Agent] = {}

    def register(self, agent: Union[Agent, AgentRecord]) -> AgentRecord:
        """
        Register a new agent with AI Brain.
        
        Args:
            agent: An instantiated Agent subclass or an AgentRecord dataclass.
            
        Returns:
            The registered AgentRecord.
            
        Raises:
            DuplicateAgentError: If an agent with the same ID is already registered.
            AgentRegistrationError: If configuration contains prohibited secrets or fields are invalid.
        """
        if isinstance(agent, Agent):
            record = agent.record
            instance = agent
        elif isinstance(agent, AgentRecord):
            record = agent
            instance = None
        else:
            raise AgentRegistrationError(f"Expected Agent or AgentRecord, got {type(agent).__name__}")

        # Check for duplicate registration in persistent store and active cache
        if self.store.get(record.agent_id) is not None:
            raise DuplicateAgentError(f"Agent with ID '{record.agent_id}' is already registered.")

        # Persist to store
        self.store.save(record)

        if instance is not None:
            self._instances[record.agent_id] = instance

        logger.info("Registered agent '%s' (%s, provider=%s)", record.agent_id, record.agent_type.value, record.provider)
        return record

    def attach_instance(self, agent: Agent) -> None:
        """Attach a live Agent instance to an already-persisted AgentRecord."""
        record = self.store.get(agent.agent_id)
        if record is None:
            raise AgentNotFoundError(f"Cannot attach instance: agent '{agent.agent_id}' is not registered.")
        self._instances[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[Agent]:
        """
        Retrieve a live Agent instance by ID.
        Returns None if not instantiated in the current session.
        """
        return self._instances.get(agent_id)

    def get_record(self, agent_id: str) -> Optional[AgentRecord]:
        """
        Retrieve an agent's persistent identity record by ID.
        """
        instance = self._instances.get(agent_id)
        if instance is not None:
            return instance.record
        return self.store.get(agent_id)

    def get_or_raise(self, agent_id: str) -> AgentRecord:
        """Retrieve an agent's record or raise AgentNotFoundError."""
        record = self.get_record(agent_id)
        if record is None:
            raise AgentNotFoundError(f"Agent '{agent_id}' is not registered in AI Brain.")
        return record

    def list(
        self,
        *,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        provider: Optional[str] = None,
    ) -> List[AgentRecord]:
        """List registered agents matching the specified criteria."""
        return self.store.list(status=status, agent_type=agent_type, provider=provider)

    def activate(self, agent_id: str) -> AgentRecord:
        """Activate an agent for processing inputs."""
        return self.update_status(agent_id, AgentStatus.ACTIVE)

    def deactivate(self, agent_id: str) -> AgentRecord:
        """Deactivate an agent, preventing it from executing interactions."""
        return self.update_status(agent_id, AgentStatus.INACTIVE)

    def update_status(self, agent_id: str, status: AgentStatus) -> AgentRecord:
        """Update an agent's lifecycle status."""
        record = self.get_or_raise(agent_id)
        updated_record = record.with_status(status)
        self.store.save(updated_record)

        # Update in-memory instance if present
        instance = self._instances.get(agent_id)
        if instance is not None:
            instance._record = updated_record

        logger.info("Updated status for agent '%s' -> %s", agent_id, status.value)
        return updated_record

    def update_configuration(self, agent_id: str, configuration: Mapping[str, Any]) -> AgentRecord:
        """
        Update an agent's configuration parameters.
        Prohibits credentials, secrets, or API keys.
        """
        _validate_no_secrets(configuration)
        record = self.get_or_raise(agent_id)
        updated_record = record.with_configuration(configuration)
        self.store.save(updated_record)

        instance = self._instances.get(agent_id)
        if instance is not None:
            instance._record = updated_record

        logger.info("Updated configuration for agent '%s'", agent_id)
        return updated_record

    def close(self) -> None:
        """Close registry storage connections."""
        self.store.close()

