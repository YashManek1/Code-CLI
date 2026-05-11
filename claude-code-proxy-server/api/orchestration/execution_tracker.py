"""Track completed work and update remaining tasks.

Maintains the execution progress within a session by moving steps
between the ``remaining_steps`` and ``completed_steps`` lists in the
persistent ExecutionState.
"""
from __future__ import annotations

from loguru import logger

from ..execution_state_store import ExecutionStateStore
from ..models.execution_state import ExecutionState, PlanStep
from .plan_parser import (
    normalize_plan,
    parse_plan_text,
    split_by_status,
)

class ExecutionTracker:
    """Stateless tracker that operates on the persistent store."""

    def __init__(self, store: ExecutionStateStore) -> None:
        self._store = store

    def mark_step_completed(
        self, session_id: str, step_id: str
    ) -> ExecutionState | None:
        """Mark a step as completed by its ``step_id``.

        Moves the step from ``remaining_steps`` to ``completed_steps``
        and persists the change.  Returns the updated state, or ``None``
        if the session or step is not found.
        """
        state = self._store.ensure_state(session_id)

        target_step: PlanStep | None = None
        for step in state.remaining_steps:
            if step.step_id == step_id:
                target_step = step
                break

        if target_step is None:
            logger.warning(
                "EXECUTION_TRACKER: step_id={} not found in remaining for session={}",
                step_id,
                session_id,
            )
            return state

        return self._store.append_completed_step(session_id, target_step)

    def mark_step_in_progress(
        self, session_id: str, step_id: str
    ) -> ExecutionState | None:
        """Mark a remaining step as in-progress."""
        state = self._store.load(session_id)
        if state is None:
            return None

        updated_remaining: list[PlanStep] = []
        for step in state.remaining_steps:
            if step.step_id == step_id:
                updated_remaining.append(step.mark_in_progress())
            else:
                updated_remaining.append(step)

        state.remaining_steps = updated_remaining
        self._store.save(state)
        return state

    def get_progress(self, session_id: str) -> tuple[int, int]:
        """Return ``(completed_count, total_count)`` for a session.

        Returns ``(0, 0)`` if the session has no state.
        """
        state = self._store.load(session_id)
        if state is None:
            return 0, 0
        return state.progress_summary()

    def get_next_step(self, session_id: str) -> PlanStep | None:
        """Return the first pending/in-progress remaining step."""
        state = self._store.load(session_id)
        if state is None:
            return None
        for step in state.remaining_steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    def apply_approved_plan(
        self,
        session_id: str,
        plan_text: str,
    ) -> ExecutionState | None:
        """Parse and persist an approved execution plan."""
        state = self._store.load(session_id)

        if state is None:
            return None

        parsed_steps = parse_plan_text(plan_text)

        parsed_steps = normalize_plan(parsed_steps)

        completed_steps, remaining_steps = split_by_status(
            parsed_steps
        )

        state.approved_plan = plan_text
        state.completed_steps = completed_steps
        state.remaining_steps = remaining_steps

        self._store.save(state)

        return state
