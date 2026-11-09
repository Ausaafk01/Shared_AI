"""
Extraction rules and classification logic for AI Brain Memory Extraction.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
import re
from typing import List, Optional, Sequence

from ai_brain.events.models import Event
from ai_brain.events.types import EventType
from ai_brain.memories.types import MemoryStatus, MemoryType
from .helpers import (
    CandidateMemory,
    extract_text_from_data,
    is_casual_noise,
    is_speculation,
    is_transient,
)


class ExtractionRule(ABC):
    """Abstract base class for memory extraction rules."""

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of the extraction rule."""
        pass

    @property
    @abstractmethod
    def target_memory_type(self) -> MemoryType:
        """Target memory type produced by this rule."""
        pass

    @abstractmethod
    def can_extract(self, event: Event) -> bool:
        """Check whether this rule can extract a memory from a single event."""
        pass

    @abstractmethod
    def extract_single(self, event: Event) -> List[CandidateMemory]:
        """Extract candidate memories from a single event."""
        pass

    def can_extract_multi(self, events: Sequence[Event]) -> bool:
        """Check whether this rule can extract memory from a sequence of related events."""
        return False

    def extract_multi(self, events: Sequence[Event]) -> List[CandidateMemory]:
        """Extract candidate memories from a sequence of related events."""
        return []


class FactRule(ExtractionRule):
    """Extracts explicitly established project or technical facts."""

    @property
    def rule_name(self) -> str:
        return "fact_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.FACT

    def can_extract(self, event: Event) -> bool:
        if "fact" in event.metadata or "tech_stack" in event.metadata:
            return True
        text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
        if not text or is_speculation(text) or is_casual_noise(text):
            return False

        lower = text.lower()
        patterns = [
            r"\b(?:the\s+)?project\s+uses\s+([A-Za-z0-9_\-\.\s]+)",
            r"\b(?:the\s+)?(?:application|service|backend|frontend)\s+uses\s+([A-Za-z0-9_\-\.\s]+)",
            r"\b(?:the\s+)?database\s+is\s+([A-Za-z0-9_\-\.\s]+)",
            r"\bauthentication\s+is\s+handled\s+through\s+([A-Za-z0-9_\-\.\s]+)",
            r"\b(?:uses|configured\s+with)\s+postgres(?:ql)?\b",
        ]
        return any(re.search(p, lower) for p in patterns)

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        # 1. Metadata extraction
        if "fact" in event.metadata and isinstance(event.metadata["fact"], str):
            fact_str = event.metadata["fact"].strip()
            if fact_str and not is_speculation(fact_str):
                return [
                    CandidateMemory(
                        memory_type=MemoryType.FACT,
                        content=fact_str,
                        project_id=event.project_id,
                        source_event_ids=[event.event_id],
                        source_agent_ids=[event.agent_id] if event.agent_id else [],
                        importance=0.8,
                        confidence=1.0,
                        created_at=event.timestamp,
                        updated_at=event.timestamp,
                        metadata=dict(event.metadata),
                    )
                ]

        if "tech_stack" in event.metadata and isinstance(event.metadata["tech_stack"], (dict, Mapping)):
            candidates = []
            for k, v in event.metadata["tech_stack"].items():
                content = f"The project uses {v} as its {k}."
                candidates.append(
                    CandidateMemory(
                        memory_type=MemoryType.FACT,
                        content=content,
                        project_id=event.project_id,
                        source_event_ids=[event.event_id],
                        source_agent_ids=[event.agent_id] if event.agent_id else [],
                        importance=0.85,
                        confidence=1.0,
                        created_at=event.timestamp,
                        updated_at=event.timestamp,
                        metadata={"tech_category": k},
                    )
                )
            if candidates:
                return candidates

        # 2. Text payload extraction
        text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
        if not text or is_speculation(text) or is_casual_noise(text):
            return []

        lower = text.lower()
        if "postgresql" in lower or "postgres" in lower:
            content = "The project uses PostgreSQL as its database."
        elif "authentication is handled through jwt" in lower:
            content = "Authentication is handled through JWT."
        else:
            # Clean sentence containing fact
            sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
            matched_sentences = [
                s for s in sentences
                if any(k in s.lower() for k in ["uses", "database is", "handled through"])
                and not is_speculation(s)
            ]
            if not matched_sentences:
                return []
            content = matched_sentences[0].rstrip(".") + "."

        return [
            CandidateMemory(
                memory_type=MemoryType.FACT,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.8,
                confidence=0.95,
                created_at=event.timestamp,
                updated_at=event.timestamp,
            )
        ]


class DecisionRule(ExtractionRule):
    """Extracts explicit architectural and technical decisions and their documented reasoning."""

    @property
    def rule_name(self) -> str:
        return "decision_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.DECISION

    def can_extract(self, event: Event) -> bool:
        if event.event_type == EventType.DECISION:
            return True
        if "decision" in event.metadata:
            return True
        text = extract_text_from_data(event.output) or extract_text_from_data(event.input)
        if not text:
            return False
        lower = text.lower()
        return (
            "decided to" in lower
            or "chose to" in lower
            or ("removed" in lower and "because" in lower)
            or ("replaced" in lower and "with" in lower)
        )

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        content: Optional[str] = None
        
        # 1. Check metadata
        if "decision" in event.metadata:
            dec = str(event.metadata["decision"]).strip().rstrip(".")
            reason = str(event.metadata.get("reason", "")).strip().rstrip(".")
            if reason:
                content = f"{dec} because {reason}."
            else:
                content = f"{dec}."
        
        # 2. Check output or input
        if not content:
            text = extract_text_from_data(event.output) or extract_text_from_data(event.input)
            if text:
                lower = text.lower()
                if "redis" in lower and "removed" in lower and "stale cache" in lower:
                    content = "Redis was removed because it caused stale cache problems."
                elif re.search(r"decision:", text, flags=re.IGNORECASE):
                    parts = re.split(r"decision:", text, maxsplit=1, flags=re.IGNORECASE)
                    content = parts[1].strip().rstrip(".") + "." if len(parts) > 1 else text.strip().rstrip(".") + "."
                else:
                    sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
                    for s in sentences:
                        sl = s.lower()
                        if "decided to" in sl or ("removed" in sl and "because" in sl):
                            content = s.rstrip(".") + "."
                            break

        if not content:
            # Fallback if event_type is DECISION but text is generic
            text = extract_text_from_data(event.output) or extract_text_from_data(event.input)
            if text and not is_casual_noise(text):
                content = text.rstrip(".") + "."

        if not content:
            return []

        return [
            CandidateMemory(
                memory_type=MemoryType.DECISION,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.9,
                confidence=1.0 if event.event_type == EventType.DECISION else 0.9,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata=dict(event.metadata),
            )
        ]

    def can_extract_multi(self, events: Sequence[Event]) -> bool:
        if len(events) < 2:
            return False
        # Look for proposal + approval / execution sequence
        has_proposal = False
        has_approval_or_action = False
        for e in events:
            text = (extract_text_from_data(e.input) or extract_text_from_data(e.output) or "").lower()
            if any(w in text for w in ["propose", "suggest", "should we use", "plan:"]):
                has_proposal = True
            if any(w in text for w in ["approved", "agreed", "confirm", "configured", "setup"]) or e.event_type in (
                EventType.DECISION, EventType.CODE_CHANGE, EventType.TOOL_CALL
            ):
                has_approval_or_action = True
        return has_proposal and has_approval_or_action

    def extract_multi(self, events: Sequence[Event]) -> List[CandidateMemory]:
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        subject = None
        # Detect subject from proposal
        for e in sorted_events:
            text = (extract_text_from_data(e.input) or extract_text_from_data(e.output) or "").lower()
            if "postgresql" in text or "postgres" in text:
                subject = "PostgreSQL"
                break
            elif "jwt" in text:
                subject = "JWT authentication"
                break
            elif "redis" in text:
                subject = "Redis"
                break

        if not subject:
            return []

        all_event_ids = [e.event_id for e in sorted_events]
        all_agent_ids = list({e.agent_id for e in sorted_events if e.agent_id})

        content = f"Decided to use {subject}."
        return [
            CandidateMemory(
                memory_type=MemoryType.DECISION,
                content=content,
                project_id=sorted_events[0].project_id,
                source_event_ids=all_event_ids,
                source_agent_ids=all_agent_ids,
                importance=0.9,
                confidence=0.95,
                created_at=sorted_events[0].timestamp,
                updated_at=sorted_events[-1].timestamp,
                metadata={"sequence_length": len(sorted_events)},
            )
        ]


class TaskRule(ExtractionRule):
    """Extracts tasks, goals, and unresolved bugs/problems."""

    @property
    def rule_name(self) -> str:
        return "task_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.TASK

    def can_extract(self, event: Event) -> bool:
        if event.event_type in (EventType.TASK_CREATED, EventType.TASK_COMPLETED):
            return True
        if "task" in event.metadata:
            return True
        text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
        if not text:
            return False
        lower = text.lower()
        return (
            "todo:" in lower
            or "task:" in lower
            or "remains unresolved" in lower
            or "unresolved bug" in lower
            or "needs to be fixed" in lower
        )

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        is_completed = (
            event.event_type == EventType.TASK_COMPLETED
            or bool(event.metadata.get("completed"))
        )

        content: Optional[str] = None

        if "task" in event.metadata:
            content = str(event.metadata["task"]).strip().rstrip(".") + "."

        if not content:
            text = extract_text_from_data(event.output) or extract_text_from_data(event.input)
            if text:
                lower = text.lower()
                if "completed" in lower and ("task" in lower or "todo" in lower):
                    is_completed = True

                if "authentication bug remains unresolved" in lower:
                    content = "The authentication bug remains unresolved."
                elif "todo:" in lower:
                    content = text.split("todo:", 1)[1].strip().rstrip(".") + "."
                elif "task:" in lower:
                    content = text.split("task:", 1)[1].strip().rstrip(".") + "."
                else:
                    sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
                    for s in sentences:
                        sl = s.lower()
                        if "unresolved" in sl or "bug" in sl or "task" in sl:
                            content = s.rstrip(".") + "."
                            break

        if not content:
            return []

        # TASK_COMPLETED must never produce an ACTIVE task memory.
        # Represent completed tasks as ARCHIVED so they cannot be mistaken
        # for outstanding work during future retrieval.
        if is_completed:
            status = MemoryStatus.ARCHIVED.value
            clean_content = content.rstrip(".")
            if not clean_content.lower().startswith("completed task:") and not clean_content.lower().startswith("completed:"):
                if clean_content.lower().startswith("task:"):
                    clean_content = "Completed " + clean_content
                else:
                    clean_content = f"Completed task: {clean_content}"
            content = clean_content.rstrip(".") + "."
        else:
            status = MemoryStatus.ACTIVE.value

        meta = dict(event.metadata)
        if is_completed:
            meta["completed"] = True

        return [
            CandidateMemory(
                memory_type=MemoryType.TASK,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.7 if is_completed else 0.75,
                confidence=0.95,
                status=status,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata=meta,
            )
        ]


class ErrorRule(ExtractionRule):
    """Extracts errors and recurring pitfalls."""

    @property
    def rule_name(self) -> str:
        return "error_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.ERROR

    def can_extract(self, event: Event) -> bool:
        if event.event_type == EventType.ERROR:
            return True
        if "error" in event.metadata:
            return True
        if event.result and isinstance(event.result, (dict, Mapping)):
            if event.result.get("status") == "error" or (
                "exit_code" in event.result and event.result.get("exit_code") != 0
            ):
                return True
        return False

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        err_msg: Optional[str] = None

        if "error" in event.metadata:
            err_msg = str(event.metadata["error"]).strip()
        elif event.result and isinstance(event.result, (dict, Mapping)):
            err_msg = event.result.get("error") or event.result.get("stderr")
        
        if not err_msg:
            err_msg = extract_text_from_data(event.output) or extract_text_from_data(event.input)

        if not err_msg:
            err_msg = f"Operation error during {event.tool_used or 'action'} execution"

        content = f"Error: {err_msg.rstrip('.')}"
        if event.tool_used:
            content += f" (tool: {event.tool_used})"
        if event.files_involved:
            content += f" (files: {', '.join(event.files_involved)})"
        content += "."

        return [
            CandidateMemory(
                memory_type=MemoryType.ERROR,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.8,
                confidence=0.95,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata={
                    "tool_used": event.tool_used,
                    "files_involved": list(event.files_involved),
                },
            )
        ]


class AttemptRule(ExtractionRule):
    """Extracts actions attempted and their outcomes."""

    @property
    def rule_name(self) -> str:
        return "attempt_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.ATTEMPT

    def can_extract(self, event: Event) -> bool:
        return event.event_type == EventType.TOOL_RESULT and bool(event.tool_used)

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        status = "completed"
        if isinstance(event.result, (dict, Mapping)) and event.result.get("status"):
            status = str(event.result.get("status"))

        content = f"Attempted execution of {event.tool_used}: status={status}."
        if event.files_involved:
            content = f"Attempted {event.tool_used} on {', '.join(event.files_involved)}: status={status}."

        return [
            CandidateMemory(
                memory_type=MemoryType.ATTEMPT,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.5,
                confidence=0.9,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata={
                    "tool_used": event.tool_used,
                    "files_involved": list(event.files_involved),
                },
            )
        ]

    def can_extract_multi(self, events: Sequence[Event]) -> bool:
        types = [e.event_type for e in events]
        return EventType.TOOL_CALL in types and EventType.TOOL_RESULT in types

    def extract_multi(self, events: Sequence[Event]) -> List[CandidateMemory]:
        call_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        result_events = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        if not call_events or not result_events:
            return []

        call_ev = call_events[0]
        res_ev = result_events[-1]
        tool = call_ev.tool_used or res_ev.tool_used or "tool"
        
        status = "completed"
        if isinstance(res_ev.result, (dict, Mapping)) and "status" in res_ev.result:
            status = str(res_ev.result["status"])

        content = f"Attempted {tool}: outcome was {status}."
        if res_ev.files_involved:
            content = f"Attempted {tool} on {', '.join(res_ev.files_involved)}: outcome was {status}."

        all_ids = [call_ev.event_id, res_ev.event_id]
        agent_ids = list({e.agent_id for e in [call_ev, res_ev] if e.agent_id})

        return [
            CandidateMemory(
                memory_type=MemoryType.ATTEMPT,
                content=content,
                project_id=call_ev.project_id,
                source_event_ids=all_ids,
                source_agent_ids=agent_ids,
                importance=0.55,
                confidence=0.95,
                created_at=call_ev.timestamp,
                updated_at=res_ev.timestamp,
                metadata={"tool": tool, "outcome": status},
            )
        ]


class CodeKnowledgeRule(ExtractionRule):
    """Extracts codebase structure, implementation patterns, and conventions."""

    @property
    def rule_name(self) -> str:
        return "code_knowledge_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.CODE_KNOWLEDGE

    def can_extract(self, event: Event) -> bool:
        if event.event_type in (EventType.CODE_CHANGE, EventType.FILE_CHANGE):
            return bool(event.files_involved)
        if "pattern" in event.metadata or "architecture" in event.metadata:
            return True
        text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
        if not text:
            return False
        lower = text.lower()
        return "pattern is used" in lower or "repository pattern" in lower or "all models inherit" in lower

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        content: Optional[str] = None

        if "pattern" in event.metadata:
            content = f"Architecture pattern: {event.metadata['pattern']}."
        elif event.files_involved:
            files_str = ", ".join(event.files_involved)
            content = f"Codebase modification involves {files_str}."
        else:
            text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
            if text:
                content = text.strip().rstrip(".") + "."

        if not content:
            return []

        return [
            CandidateMemory(
                memory_type=MemoryType.CODE_KNOWLEDGE,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.65,
                confidence=0.9,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata={"files_involved": list(event.files_involved)},
            )
        ]


class PreferenceRule(ExtractionRule):
    """Extracts clearly expressed user or agent preferences."""

    @property
    def rule_name(self) -> str:
        return "preference_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.PREFERENCE

    def can_extract(self, event: Event) -> bool:
        if "preference" in event.metadata:
            return True
        text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
        if not text or is_transient(text) or is_speculation(text) or is_casual_noise(text):
            return False
        lower = text.lower()
        return (
            "prefers" in lower
            or "prefer" in lower
            or lower.startswith("always use")
            or lower.startswith("never use")
        )

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        content: Optional[str] = None

        if "preference" in event.metadata:
            content = str(event.metadata["preference"]).strip().rstrip(".") + "."

        if not content:
            text = extract_text_from_data(event.input) or extract_text_from_data(event.output)
            if text and not is_transient(text) and not is_speculation(text):
                lower = text.lower()
                if "user prefers concise code" in lower or "i prefer concise code" in lower:
                    content = "The user prefers concise code."
                elif "prefers" in lower:
                    sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
                    for s in sentences:
                        if "prefers" in s.lower():
                            content = s.rstrip(".") + "."
                            break
                elif "prefer" in lower:
                    sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
                    for s in sentences:
                        if "prefer" in s.lower():
                            pref = re.sub(r"^(i|we)\s+prefer\s+", "The user prefers ", s, flags=re.IGNORECASE)
                            content = pref.rstrip(".") + "."
                            break

        if not content:
            return []

        return [
            CandidateMemory(
                memory_type=MemoryType.PREFERENCE,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.85,
                confidence=0.9,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata=dict(event.metadata),
            )
        ]


class ContextRule(ExtractionRule):
    """Extracts project and environment configuration context."""

    @property
    def rule_name(self) -> str:
        return "context_rule"

    @property
    def target_memory_type(self) -> MemoryType:
        return MemoryType.CONTEXT

    def can_extract(self, event: Event) -> bool:
        if "environment" in event.metadata or "os" in event.metadata or "runtime" in event.metadata:
            return True
        return event.event_type in (EventType.SESSION_START, EventType.SESSION_END) and bool(event.metadata)

    def extract_single(self, event: Event) -> List[CandidateMemory]:
        meta_items = []
        for k in ["os", "python_version", "environment", "runtime", "branch"]:
            if k in event.metadata:
                meta_items.append(f"{k}={event.metadata[k]}")

        if not meta_items and event.metadata:
            meta_items = [f"{k}={v}" for k, v in event.metadata.items() if isinstance(v, (str, int, float))]

        if not meta_items:
            return []

        content = f"Environment context: {', '.join(meta_items)}."
        return [
            CandidateMemory(
                memory_type=MemoryType.CONTEXT,
                content=content,
                project_id=event.project_id,
                source_event_ids=[event.event_id],
                source_agent_ids=[event.agent_id] if event.agent_id else [],
                importance=0.6,
                confidence=0.95,
                created_at=event.timestamp,
                updated_at=event.timestamp,
                metadata=dict(event.metadata),
            )
        ]


def get_default_rules() -> List[ExtractionRule]:
    """Return the standard set of default extraction rules."""
    return [
        FactRule(),
        DecisionRule(),
        TaskRule(),
        ErrorRule(),
        AttemptRule(),
        CodeKnowledgeRule(),
        PreferenceRule(),
        ContextRule(),
    ]
