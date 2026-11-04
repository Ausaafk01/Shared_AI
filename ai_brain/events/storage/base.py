"""
Abstract base class for persistent Event Storage.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

from ..models import Event
from ..types import EventType


class EventStore(ABC):
    """Abstract interface defining required persistent event operations."""

    @abstractmethod
    def store_event(self, event: Event) -> None:
        """
        Persist an event to storage in an append-only manner.
        
        Args:
            event: The immutable Event to store.
            
        Raises:
            DuplicateEventError: If an event with the same event_id already exists.
            StorageError: If the persistence operation fails.
        """
        pass

    def store(self, event: Event) -> None:
        """Convenience alias for store_event."""
        self.store_event(event)

    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """
        Retrieve an event by its unique ID.
        
        Args:
            event_id: Unique event identifier.
            
        Returns:
            The Event if found, or None.
            
        Raises:
            StorageError: If reading from storage fails.
        """
        pass

    def get(self, event_id: str) -> Optional[Event]:
        """Convenience alias for get_event_by_id."""
        return self.get_event_by_id(event_id)

    @abstractmethod
    def get_events_by_project(
        self, project_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events for a project ordered chronologically."""
        pass

    @abstractmethod
    def get_events_by_agent(
        self, agent_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events involving a specific agent ordered chronologically."""
        pass

    @abstractmethod
    def get_events_by_session(
        self, session_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events in a session ordered chronologically."""
        pass

    @abstractmethod
    def get_events_by_type(
        self, event_type: EventType | str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events of a specific type ordered chronologically."""
        pass

    @abstractmethod
    def get_all_events(self, ascending: bool = True) -> list[Event]:
        """Retrieve all events in chronological order."""
        pass

    @abstractmethod
    def filter_events(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[EventType | str] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
    ) -> list[Event]:
        """Query events matching multiple criteria ordered chronologically."""
        pass

    @abstractmethod
    def count(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[EventType | str] = None,
    ) -> int:
        """Count the number of stored events matching the given criteria."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage resources."""
        pass

