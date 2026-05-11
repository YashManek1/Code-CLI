"""Tests for orchestration modules: plan parser, state injector, tracker, checkpoint."""

import json

import pytest

from api.execution_state_store import ExecutionStateStore
from api.models.execution_state import (
    CheckpointState,
    ExecutionPhase,
    ExecutionState,
    PlanStep,
)
from api.orchestration.checkpoint_manager import CheckpointManager
from api.orchestration.execution_tracker import ExecutionTracker
from api.orchestration.plan_parser import (
    normalize_plan,
    parse_plan_text,
    split_by_status,
)
from api.orchestration.state_injector import (
    build_orchestration_context,
    inject_execution_state_context,
    inject_execution_state_context_anthropic,
)


# =========================================================================
# Plan Parser
# =========================================================================


class TestPlanParser:
    def test_parse_checkbox_list(self):
        plan = """
- [ ] Write unit tests
- [x] Create models
- [ ] Implement API routes
- [/] Fix database migration
"""
        steps = parse_plan_text(plan)
        assert len(steps) == 4
        assert steps[0].description == "Write unit tests"
        assert steps[0].status == "pending"
        assert steps[1].description == "Create models"
        assert steps[1].status == "completed"
        assert steps[3].description == "Fix database migration"
        assert steps[3].status == "in_progress"

    def test_parse_numbered_list(self):
        plan = """
1. Design the schema
2. Implement the API
3. Write tests
"""
        steps = parse_plan_text(plan)
        assert len(steps) == 3
        assert steps[0].description == "Design the schema"
        assert all(s.status == "pending" for s in steps)

    def test_parse_prefers_checkboxes_over_numbers(self):
        plan = """
- [ ] Checkbox step
1. Numbered step
"""
        steps = parse_plan_text(plan)
        # Should only get checkbox step since checkboxes take priority
        assert len(steps) == 1
        assert steps[0].description == "Checkbox step"

    def test_parse_empty(self):
        steps = parse_plan_text("")
        assert steps == []

    def test_deterministic_ids(self):
        plan = "- [ ] Same step"
        steps1 = parse_plan_text(plan)
        steps2 = parse_plan_text(plan)
        assert steps1[0].step_id == steps2[0].step_id

    def test_normalize_deduplicates(self):
        steps = [
            PlanStep(step_id="a", description="x"),
            PlanStep(step_id="a", description="x duplicate"),
            PlanStep(step_id="b", description="y"),
        ]
        result = normalize_plan(steps)
        assert len(result) == 2

    def test_split_by_status(self):
        steps = [
            PlanStep(step_id="1", description="a", status="completed"),
            PlanStep(step_id="2", description="b", status="pending"),
            PlanStep(step_id="3", description="c", status="completed"),
            PlanStep(step_id="4", description="d", status="in_progress"),
        ]
        completed, remaining = split_by_status(steps)
        assert len(completed) == 2
        assert len(remaining) == 2


# =========================================================================
# State Injector
# =========================================================================


class TestBuildOrchestrationContext:
    def test_idle_state_returns_empty(self):
        state = ExecutionState(session_id="idle")
        assert build_orchestration_context(state) == ""

    def test_active_state_returns_block(self):
        state = ExecutionState(
            session_id="active",
            implementation_phase=ExecutionPhase.backend_execution,
            current_checkpoint=CheckpointState(name="Test Checkpoint"),
            completed_steps=[
                PlanStep(step_id="1", description="Step 1", status="completed"),
            ],
            remaining_steps=[
                PlanStep(step_id="2", description="Step 2"),
            ],
        )
        context = build_orchestration_context(state)
        assert "<execution_state>" in context
        assert "</execution_state>" in context
        assert "CHECKPOINT: Test Checkpoint" in context
        assert "PHASE: backend_execution" in context
        assert "[x] Step 1" in context
        assert "[ ] Step 2" in context

    def test_locked_plan_shows_summary(self):
        state = ExecutionState(
            session_id="plan",
            implementation_phase=ExecutionPhase.backend_planning,
            approved_plan="# My Big Plan\n- [ ] Step 1\n- [ ] Step 2",
        )
        context = build_orchestration_context(state)
        assert "PLAN_STATUS: locked" in context
        assert "PLAN_SUMMARY:" in context

    def test_rules_included(self):
        state = ExecutionState(
            session_id="rules",
            implementation_phase=ExecutionPhase.backend_execution,
            locked_rules=["Do not redesign", "Continue from checkpoint"],
        )
        context = build_orchestration_context(state)
        assert "RULES:" in context
        assert "Do not redesign" in context


class TestInjectOpenAI:
    def test_inject_into_empty_body(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = inject_execution_state_context(body, state)
        # Should prepend a system message
        assert result["messages"][0]["role"] == "system"
        assert "<execution_state>" in result["messages"][0]["content"]

    def test_inject_into_existing_system_message(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = inject_execution_state_context(body, state)
        # System message should contain both original and injected content
        assert "You are a helpful assistant." in result["messages"][0]["content"]
        assert "<execution_state>" in result["messages"][0]["content"]
        assert len(result["messages"]) == 2

    def test_no_injection_when_state_is_none(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = inject_execution_state_context(body, None)
        assert len(result["messages"]) == 1

    def test_idempotent_injection(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {
            "messages": [
                {"role": "system", "content": "Original system prompt"},
                {"role": "user", "content": "Hello"},
            ]
        }
        # Inject twice
        result = inject_execution_state_context(body, state)
        result = inject_execution_state_context(result, state)
        # Should only have one execution_state block
        content = result["messages"][0]["content"]
        assert content.count("<execution_state>") == 1


class TestInjectAnthropic:
    def test_inject_into_none_system(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {"messages": []}
        result = inject_execution_state_context_anthropic(body, state)
        assert isinstance(result["system"], str)
        assert "<execution_state>" in result["system"]

    def test_inject_into_string_system(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {"system": "You are helpful.", "messages": []}
        result = inject_execution_state_context_anthropic(body, state)
        assert "You are helpful." in result["system"]
        assert "<execution_state>" in result["system"]

    def test_inject_into_list_system(self):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
        )
        body = {
            "system": [{"type": "text", "text": "Original system"}],
            "messages": [],
        }
        result = inject_execution_state_context_anthropic(body, state)
        assert isinstance(result["system"], list)
        assert len(result["system"]) == 2
        assert "<execution_state>" in result["system"][0]["text"]


# =========================================================================
# Execution Tracker
# =========================================================================


@pytest.fixture
def tracker_store(tmp_path):
    return ExecutionStateStore(base_dir=str(tmp_path / "tracker_states"))


class TestExecutionTracker:
    def test_mark_step_completed(self, tracker_store):
        step = PlanStep(step_id="s1", description="Write tests")
        state = ExecutionState(
            session_id="t1",
            remaining_steps=[step],
        )
        tracker_store.save(state)

        tracker = ExecutionTracker(tracker_store)
        updated = tracker.mark_step_completed("t1", "s1")
        assert updated is not None
        assert len(updated.completed_steps) == 1
        assert len(updated.remaining_steps) == 0

    def test_mark_step_in_progress(self, tracker_store):
        step = PlanStep(step_id="s1", description="Write tests")
        state = ExecutionState(
            session_id="t1",
            remaining_steps=[step],
        )
        tracker_store.save(state)

        tracker = ExecutionTracker(tracker_store)
        updated = tracker.mark_step_in_progress("t1", "s1")
        assert updated is not None
        assert updated.remaining_steps[0].status == "in_progress"

    def test_get_progress(self, tracker_store):
        state = ExecutionState(
            session_id="t1",
            completed_steps=[
                PlanStep(step_id="1", description="a", status="completed"),
            ],
            remaining_steps=[
                PlanStep(step_id="2", description="b"),
                PlanStep(step_id="3", description="c"),
            ],
        )
        tracker_store.save(state)

        tracker = ExecutionTracker(tracker_store)
        completed, total = tracker.get_progress("t1")
        assert completed == 1
        assert total == 3

    def test_get_next_step(self, tracker_store):
        state = ExecutionState(
            session_id="t1",
            remaining_steps=[
                PlanStep(step_id="1", description="First"),
                PlanStep(step_id="2", description="Second"),
            ],
        )
        tracker_store.save(state)

        tracker = ExecutionTracker(tracker_store)
        next_step = tracker.get_next_step("t1")
        assert next_step is not None
        assert next_step.description == "First"


# =========================================================================
# Checkpoint Manager
# =========================================================================


class TestCheckpointManager:
    def test_create_checkpoint(self, tracker_store):
        mgr = CheckpointManager(tracker_store)
        state = mgr.create_checkpoint("s1", "My Checkpoint", "Description")
        assert state.current_checkpoint is not None
        assert state.current_checkpoint.name == "My Checkpoint"

    def test_restore_checkpoint(self, tracker_store):
        state = ExecutionState(
            session_id="s1",
            implementation_phase=ExecutionPhase.backend_execution,
            current_checkpoint=CheckpointState(
                name="cp1",
                phase=ExecutionPhase.backend_planning,
            ),
        )
        tracker_store.save(state)

        mgr = CheckpointManager(tracker_store)
        restored = mgr.restore_checkpoint("s1")
        assert restored is not None
        assert restored.implementation_phase == ExecutionPhase.backend_planning

    def test_advance_phase(self, tracker_store):
        tracker_store.save(ExecutionState(session_id="s1"))
        mgr = CheckpointManager(tracker_store)
        updated = mgr.advance_phase("s1", ExecutionPhase.frontend_execution)
        assert updated.implementation_phase == ExecutionPhase.frontend_execution

    def test_restore_no_checkpoint_returns_none(self, tracker_store):
        tracker_store.save(ExecutionState(session_id="s1"))
        mgr = CheckpointManager(tracker_store)
        result = mgr.restore_checkpoint("s1")
        assert result is None
