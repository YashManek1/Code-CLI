"""Manage execution checkpoints and phase transitions.

Checkpoints capture a named snapshot of execution progress so that
model switches and server restarts can resume from the last known
good position.
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from ..execution_state_store import ExecutionStateStore
from ..models.execution_state import (
    CheckpointState,
    ExecutionPhase,
    ExecutionState,
    ExecutionStateUpdate,
)


class CheckpointManager:
    """Create, restore, and advance execution checkpoints."""

    VALID_PHASES = list(ExecutionPhase)

    def __init__(self, store: ExecutionStateStore) -> None:
        self._store = store

    def create_checkpoint(
        self,
        session_id: str,
        name: str,
        description: str = "",
        phase: ExecutionPhase | None = None,
    ) -> ExecutionState:
        """Create or overwrite the active checkpoint for a session.

        If no ``phase`` is supplied, the current ``implementation_phase``
        of the session state is used.
        """
        state = self._store.ensure_state(session_id)
        effective_phase = phase if phase is not None else state.implementation_phase

        checkpoint = CheckpointState(
            name=name,
            description=description,
            phase=effective_phase,
            created_at=datetime.now(timezone.utc),
        )

        updated = self._store.update(
            session_id,
            ExecutionStateUpdate(current_checkpoint=checkpoint),
        )

        logger.info(
            "CHECKPOINT_CREATE: session={} name={} phase={}",
            session_id,
            name,
            effective_phase,
        )
        return updated

    def restore_checkpoint(
        self, session_id: str
    ) -> ExecutionState | None:
        """Restore the active checkpoint: set the implementation phase
        to the checkpoint's phase and return the state.

        Returns ``None`` if no checkpoint exists.
        """
        state = self._store.load(session_id)
        if state is None or state.current_checkpoint is None:
            logger.info(
                "CHECKPOINT_RESTORE: no checkpoint for session={}", session_id
            )
            return None

        checkpoint_phase = state.current_checkpoint.phase
        updated = self._store.update(
            session_id,
            ExecutionStateUpdate(implementation_phase=checkpoint_phase),
        )

        logger.info(
            "CHECKPOINT_RESTORE: session={} name={} phase={}",
            session_id,
            state.current_checkpoint.name,
            checkpoint_phase,
        )
        return updated

    def advance_phase(
        self, session_id: str, phase: ExecutionPhase
    ) -> ExecutionState:
        """Set the implementation phase for a session."""
        updated = self._store.update(
            session_id,
            ExecutionStateUpdate(implementation_phase=phase),
        )
        logger.info(
            "PHASE_ADVANCE: session={} phase={}", session_id, phase
        )
        return updated

    def get_current_phase(self, session_id: str) -> ExecutionPhase:
        """Return the current phase, defaulting to ``idle``."""
        state = self._store.load(session_id)
        if state is None:
            return ExecutionPhase.idle
        return state.implementation_phase
