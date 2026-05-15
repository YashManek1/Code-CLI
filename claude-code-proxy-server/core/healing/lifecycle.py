"""Autonomous session lifecycle management and context sustainability."""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from loguru import logger
from .snapshots import SnapshotManager
from api.models.execution_state import ExecutionState

class SessionLifecycleManager:
    """Manages the sustainability of long-horizon autonomous sessions."""

    def __init__(self, snapshot_manager: SnapshotManager):
        self.snapshot_manager = snapshot_manager
        self._start_time = time.time()
        self._last_rebase = self._start_time

    def check_and_rebase(self, state: ExecutionState) -> bool:
        """Determine if a memory rebase is needed and execute it."""
        now = time.time()
        
        # Rebase every 30 minutes or if the history is too large
        if now - self._last_rebase > 1800 or len(state.completed_steps) > 20:
            logger.info("LIFECYCLE: Initiating memory rebase for session {}", state.session_id)
            self._execute_rebase(state)
            self._last_rebase = now
            return True
        return False

    def _execute_rebase(self, state: ExecutionState) -> None:
        """Compress history into high-density summaries."""
        # 1. Compress completed steps
        if len(state.completed_steps) > 5:
            summary = f"COMPACTION: Completed {len(state.completed_steps)} steps. "
            # We keep the last 3 steps in full, compress the rest
            retained = state.completed_steps[-3:]
            # In a real system, we'd generate a semantic summary here
            state.completed_steps = retained
            state.validation_findings.append(f"{summary} History compressed to save tokens.")

        # 2. Prune old validation failures
        if len(state.validation_failures) > 10:
            state.validation_failures = state.validation_failures[-5:]

    def evict_stale_context(self, state: ExecutionState) -> None:
        """Remove files or logs that are no longer relevant to the current phase."""
        if len(state.active_files) > 10:
            # Simple heuristic: keep the 10 most recently touched files
            state.active_files = state.active_files[-10:]
            logger.info("LIFECYCLE: Evicted stale files from active context.")

class StabilityAnalytics:
    """Telemetry system for tracking autonomous stability KPIs."""

    def __init__(self):
        self.metrics = {
            "hallucination_count": 0,
            "repair_success_count": 0,
            "total_failures": 0,
            "start_time": time.time()
        }

    def record_failure(self, failure_type: str):
        self.metrics["total_failures"] += 1
        if failure_type == "recursive_repair_detected":
            self.metrics["hallucination_count"] += 1

    def record_success(self):
        self.metrics["repair_success_count"] += 1

    def record_event(self, event_type: str):
        """Record various stability events for KPI tracking."""
        if event_type == "stream_interruptions":
            self.metrics["total_failures"] += 1
        elif event_type == "successful_healing":
            self.metrics["repair_success_count"] += 1
        elif event_type == "hallucination":
            self.metrics["hallucination_count"] += 1

    def get_kpis(self) -> Dict[str, Any]:
        """Return standardized stability metrics."""
        return {
            "total_interruptions": self.metrics["total_failures"],
            "healing_success_rate": self.get_hallucination_rate(), # Mocking for now
            "mtbf_seconds": self.get_mtbf(),
            "hallucination_rate": self.get_hallucination_rate()
        }

    def get_mtbf(self) -> float:
        """Mean Time Between Failures in seconds."""
        elapsed = time.time() - self.metrics["start_time"]
        if self.metrics["total_failures"] == 0:
            return elapsed
        return elapsed / self.metrics["total_failures"]

    def get_hallucination_rate(self) -> float:
        if self.metrics["total_failures"] == 0:
            return 0.0
        return self.metrics["hallucination_count"] / self.metrics["total_failures"]
