"""Tests for ExecutionState models, store, and serialization."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from api.execution_state_store import ExecutionStateStore
from api.models.execution_state import (
    CheckpointState,
    ExecutionPhase,
    ExecutionState,
    ExecutionStateUpdate,
    PlanStep,
)


# =========================================================================
# Model tests
# =========================================================================


class TestExecutionPhase:
    def test_all_phases_are_strings(self):
        for phase in ExecutionPhase:
            assert isinstance(phase.value, str)

    def test_idle_is_default(self):
        state = ExecutionState(session_id="test")
        assert state.implementation_phase == ExecutionPhase.idle


class TestPlanStep:
    def test_mark_completed(self):
        step = PlanStep(step_id="s1", description="Write tests")
        completed = step.mark_completed()
        assert completed.status == "completed"
        assert completed.completed_at is not None
        # Original unchanged
        assert step.status == "pending"

    def test_mark_in_progress(self):
        step = PlanStep(step_id="s1", description="Write tests")
        in_progress = step.mark_in_progress()
        assert in_progress.status == "in_progress"


class TestCheckpointState:
    def test_defaults(self):
        cp = CheckpointState(name="v1")
        assert cp.phase == ExecutionPhase.idle
        assert cp.description == ""
        assert cp.created_at is not None


class TestExecutionState:
    def test_defaults(self):
        state = ExecutionState(session_id="s1")
        assert state.active_model == ""
        assert state.current_checkpoint is None
        assert state.implementation_phase == ExecutionPhase.idle
        assert state.approved_plan is None
        assert state.completed_steps == []
        assert state.remaining_steps == []
        assert state.version == 1

    def test_progress_summary(self):
        state = ExecutionState(
            session_id="s1",
            completed_steps=[
                PlanStep(step_id="1", description="a", status="completed"),
                PlanStep(step_id="2", description="b", status="completed"),
            ],
            remaining_steps=[
                PlanStep(step_id="3", description="c"),
            ],
        )
        completed, total = state.progress_summary()
        assert completed == 2
        assert total == 3

    def test_has_active_plan(self):
        state = ExecutionState(session_id="s1")
        assert not state.has_active_plan()

        state.approved_plan = "# Plan"
        assert not state.has_active_plan()

        state.remaining_steps = [PlanStep(step_id="1", description="do something")]
        assert state.has_active_plan()

    def test_touch(self):
        state = ExecutionState(session_id="s1")
        old_ts = state.last_updated
        state.touch()
        assert state.last_updated >= old_ts

    def test_serialization_roundtrip(self):
        state = ExecutionState(
            session_id="test-123",
            active_model="gpt-4",
            implementation_phase=ExecutionPhase.backend_execution,
            approved_plan="# My Plan\n- [ ] step 1",
            current_checkpoint=CheckpointState(
                name="cp1", description="First checkpoint"
            ),
            completed_steps=[
                PlanStep(step_id="s1", description="done", status="completed")
            ],
            remaining_steps=[
                PlanStep(step_id="s2", description="todo", status="pending")
            ],
            locked_rules=["Do not redesign architecture"],
            active_files=["main.py"],
        )
        dumped = state.model_dump(mode="json")
        raw = json.dumps(dumped)
        loaded = json.loads(raw)
        restored = ExecutionState.model_validate(loaded)
        assert restored.session_id == "test-123"
        assert restored.implementation_phase == ExecutionPhase.backend_execution
        assert restored.current_checkpoint is not None
        assert restored.current_checkpoint.name == "cp1"
        assert len(restored.completed_steps) == 1
        assert len(restored.remaining_steps) == 1
        assert restored.locked_rules == ["Do not redesign architecture"]


# =========================================================================
# Store tests
# =========================================================================


@pytest.fixture
def temp_store(tmp_path: Path):
    """Return an ExecutionStateStore backed by a temporary directory."""
    return ExecutionStateStore(base_dir=str(tmp_path / "states"))


class TestExecutionStateStore:
    def test_load_nonexistent_returns_none(self, temp_store: ExecutionStateStore):
        assert temp_store.load("nonexistent") is None

    def test_save_and_load(self, temp_store: ExecutionStateStore):
        state = ExecutionState(
            session_id="s1",
            active_model="test-model",
            implementation_phase=ExecutionPhase.backend_planning,
        )
        temp_store.save(state)
        loaded = temp_store.load("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.active_model == "test-model"
        assert loaded.implementation_phase == ExecutionPhase.backend_planning

    def test_update_creates_new(self, temp_store: ExecutionStateStore):
        patch = ExecutionStateUpdate(
            active_model="new-model",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        updated = temp_store.update("new-session", patch)
        assert updated.session_id == "new-session"
        assert updated.active_model == "new-model"
        assert updated.implementation_phase == ExecutionPhase.backend_execution

    def test_update_preserves_existing_fields(self, temp_store: ExecutionStateStore):
        state = ExecutionState(
            session_id="s1",
            active_model="model-a",
            approved_plan="# Plan",
        )
        temp_store.save(state)

        patch = ExecutionStateUpdate(active_model="model-b")
        updated = temp_store.update("s1", patch)
        assert updated.active_model == "model-b"
        assert updated.approved_plan == "# Plan"  # Preserved

    def test_append_completed_step(self, temp_store: ExecutionStateStore):
        step = PlanStep(step_id="s1", description="Write tests")
        state = ExecutionState(
            session_id="session1",
            remaining_steps=[step],
        )
        temp_store.save(state)

        updated = temp_store.append_completed_step("session1", step)
        assert updated is not None
        assert len(updated.completed_steps) == 1
        assert len(updated.remaining_steps) == 0
        assert updated.completed_steps[0].status == "completed"

    def test_append_completed_step_idempotent(self, temp_store: ExecutionStateStore):
        step = PlanStep(step_id="s1", description="Write tests")
        state = ExecutionState(
            session_id="session1",
            remaining_steps=[step],
        )
        temp_store.save(state)

        temp_store.append_completed_step("session1", step)
        temp_store.append_completed_step("session1", step)

        loaded = temp_store.load("session1")
        assert loaded is not None
        assert len(loaded.completed_steps) == 1

    def test_ensure_state_creates_if_missing(self, temp_store: ExecutionStateStore):
        state = temp_store.ensure_state("new-session")
        assert state.session_id == "new-session"
        # Verify persisted
        loaded = temp_store.load("new-session")
        assert loaded is not None

    def test_delete(self, temp_store: ExecutionStateStore):
        state = ExecutionState(session_id="to-delete")
        temp_store.save(state)
        assert temp_store.load("to-delete") is not None

        deleted = temp_store.delete("to-delete")
        assert deleted is True
        assert temp_store.load("to-delete") is None

    def test_list_sessions(self, temp_store: ExecutionStateStore):
        for sid in ["a", "b", "c"]:
            temp_store.save(ExecutionState(session_id=sid))
        sessions = temp_store.list_sessions()
        assert set(sessions) == {"a", "b", "c"}

    def test_corrupt_file_returns_none(self, temp_store: ExecutionStateStore):
        # Write corrupt JSON
        path = temp_store._file_path("corrupt")
        path.write_text("not valid json!!!", encoding="utf-8")
        result = temp_store.load("corrupt")
        assert result is None

    def test_atomic_write_survives(self, temp_store: ExecutionStateStore):
        state = ExecutionState(session_id="atomic-test", active_model="m1")
        temp_store.save(state)
        # Overwrite
        state.active_model = "m2"
        temp_store.save(state)
        loaded = temp_store.load("atomic-test")
        assert loaded is not None
        assert loaded.active_model == "m2"
