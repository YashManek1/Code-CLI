"""Thread-safe JSON file persistence for ExecutionState.

Storage layout:
    .execution_states/{session_id}.json

Design choices:
- Atomic writes via temp-file + os.replace to prevent corruption.
- Per-session file locks to avoid contention across sessions.
- Graceful corruption handling: malformed files are logged and ignored.
- No database dependency — plain JSON for simplicity and portability.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from .models.execution_state import (
    ExecutionState,
    ExecutionStateUpdate,
    PlanStep,
)


class ExecutionStateStore:
    """Persistent store for execution state backed by JSON files."""

    def __init__(self, base_dir: str = ".execution_states") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _session_lock(self, session_id: str) -> threading.Lock:
        """Return or create a per-session lock."""
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def _file_path(self, session_id: str) -> Path:
        """Return the JSON file path for a session."""
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"{safe_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> ExecutionState | None:
        """Load execution state from disk.

        Returns None if the file doesn't exist or is corrupt.
        """
        path = self._file_path(session_id)
        if not path.is_file():
            return None

        lock = self._session_lock(session_id)
        with lock:
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                return ExecutionState.model_validate(data)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                logger.warning(
                    "EXECUTION_STATE_LOAD: corrupt state file for session={}: {}",
                    session_id,
                    type(exc).__name__,
                )
                return None

    def save(self, state: ExecutionState) -> None:
        """Persist execution state atomically."""
        state.touch()
        path = self._file_path(state.session_id)
        lock = self._session_lock(state.session_id)

        serialized = state.model_dump(mode="json")
        payload = json.dumps(serialized, indent=2, ensure_ascii=False)

        with lock:
            self._atomic_write(path, payload)

        logger.debug(
            "EXECUTION_STATE_SAVE: session={} phase={} steps={}/{}",
            state.session_id,
            state.implementation_phase,
            *state.progress_summary(),
        )

    def update(
        self, session_id: str, patch: ExecutionStateUpdate
    ) -> ExecutionState:
        """Apply a partial update to an existing state (or create a new one).

        Only non-None fields in ``patch`` are applied.
        """
        existing = self.load(session_id)
        if existing is None:
            existing = ExecutionState(session_id=session_id)

        update_data: dict[str, Any] = {}
        for field_name in patch.model_fields_set:
            field_value = getattr(patch, field_name)
            if field_value is not None:
                update_data[field_name] = field_value

        updated = existing.model_copy(update=update_data) if update_data else existing

        self.save(updated)
        return updated

    def append_completed_step(
        self, session_id: str, step: PlanStep
    ) -> ExecutionState | None:
        """Mark a step as completed and move it from remaining to completed.

        Returns the updated state or None if session doesn't exist.
        """
        lock = self._session_lock(session_id)
        with lock:
            state = self._load_unlocked(session_id)
            if state is None:
                return None

            completed_step = step.mark_completed()

            # Remove from remaining if present
            state.remaining_steps = [
                s for s in state.remaining_steps if s.step_id != step.step_id
            ]

            # Add to completed if not already there
            existing_ids = {s.step_id for s in state.completed_steps}
            if completed_step.step_id not in existing_ids:
                state.completed_steps.append(completed_step)

            state.touch()
            path = self._file_path(session_id)
            serialized = state.model_dump(mode="json")
            payload = json.dumps(serialized, indent=2, ensure_ascii=False)
            self._atomic_write(path, payload)

            return state

    def ensure_state(self, session_id: str) -> ExecutionState:
        """Load existing state or create a new empty one."""
        state = self.load(session_id)
        if state is None:
            state = ExecutionState(session_id=session_id)
            self.save(state)
        return state

    def ensure_state_from_parent(
        self, session_id: str, parent_session_id: str | None
    ) -> ExecutionState:
        """Load/create state, cloning parent orchestration state when needed.

        Claude Code can intentionally regenerate the CLI session id when plan
        mode exits with a context clear.  In that flow, the implementation
        session should inherit the locked plan and checkpoint from the planning
        session instead of starting from an empty orchestration state.
        """
        state = self.load(session_id)
        if state is not None:
            if parent_session_id and state.parent_session_id != parent_session_id:
                state.parent_session_id = parent_session_id
                self.save(state)
            return state

        parent = self.load(parent_session_id) if parent_session_id else None
        if parent is not None:
            state = parent.model_copy(
                deep=True,
                update={
                    "session_id": session_id,
                    "parent_session_id": parent_session_id,
                },
            )
        else:
            state = ExecutionState(
                session_id=session_id,
                parent_session_id=parent_session_id,
            )
        self.save(state)
        return state

    def delete(self, session_id: str) -> bool:
        """Delete the state file for a session. Returns True if deleted."""
        path = self._file_path(session_id)
        lock = self._session_lock(session_id)
        with lock:
            try:
                path.unlink(missing_ok=True)
                return True
            except OSError as exc:
                logger.warning(
                    "EXECUTION_STATE_DELETE: failed for session={}: {}",
                    session_id,
                    type(exc).__name__,
                )
                return False

    def list_sessions(self) -> list[str]:
        """Return all session IDs with persisted state."""
        sessions: list[str] = []
        with suppress(OSError):
            sessions.extend(path.stem for path in self._base_dir.glob("*.json"))
        return sessions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_unlocked(self, session_id: str) -> ExecutionState | None:
        """Load without acquiring the lock (caller must hold it)."""
        path = self._file_path(session_id)
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return ExecutionState.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def _atomic_write(self, path: Path, payload: str) -> None:
        """Write ``payload`` to ``path`` atomically via temp-file + rename."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._base_dir),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            with suppress(OSError):
                os.unlink(tmp_path)
            raise
