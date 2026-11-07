"""Test synchronization agent for deterministic worker crash-recovery verification.

Exclusively activated when explicitly armed by the test harness via thread event
or container filesystem barrier. In normal/production operation without the test
barrier, executes immediately with zero blocking or delay.
"""

from __future__ import annotations

import os
import time
from typing import Optional
import threading

from ai_brain.agents.base import Agent
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import InteractionContextResult


class TestSyncBlockingAgent(Agent):
    """
    Test-only agent that blocks deterministically during execution IF AND ONLY IF
    a test synchronization barrier is armed on the filesystem or via thread events.
    In normal/production environments without the test barrier, it executes immediately.
    """

    BARRIER_FILE = "/tmp/ai_brain_test_sync_barrier"
    PROCESSING_FILE = "/tmp/ai_brain_test_sync_processing"
    _thread_started_event: Optional[threading.Event] = None
    _thread_barrier_event: Optional[threading.Event] = None

    def __init__(self, agent_id: str = "test-sync-blocker") -> None:
        record = AgentRecord(
            agent_id=agent_id,
            name="Test Sync Blocking Agent",
            provider="test-sync",
            agent_type=AgentType.CODING,
            capabilities=(
                AgentCapability.CODE_DEVELOPMENT.value,
                AgentCapability.ARCHITECTURE_DESIGN.value,
            ),
            configuration={},
            status=AgentStatus.ACTIVE,
        )
        super().__init__(record)

    def generate_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> str:
        # 1. Thread synchronization event (for in-process unit/integration test harness)
        if self._thread_started_event is not None:
            self._thread_started_event.set()
        if self._thread_barrier_event is not None:
            self._thread_barrier_event.wait(timeout=30.0)

        # 2. Filesystem barrier (for containerized Docker acceptance test harness)
        if os.path.exists(self.BARRIER_FILE):
            try:
                with open(self.PROCESSING_FILE, "w", encoding="utf-8") as f:
                    f.write(f"{os.getpid()}:{time.time()}\n")
            except Exception:
                pass

            start_wait = time.time()
            # Block deterministically until the test harness removes the barrier or terminates process
            while os.path.exists(self.BARRIER_FILE) and time.time() - start_wait < 30.0:
                time.sleep(0.05)

        return (
            "Verified architectural decision: We decided to deploy PostgreSQL with automated "
            "lease recovery and crash-resilient worker execution."
        )
