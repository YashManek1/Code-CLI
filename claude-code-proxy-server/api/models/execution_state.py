"""Strongly typed models for persistent execution-state orchestration.

These models represent the complete orchestration state that survives
model switches, session resumes, and server restarts.  They are the
single source of truth consumed by the state store, the context
injector, and the CLI sync layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ExecutionPhase(StrEnum):
    """Lifecycle phases for implementation execution."""

    idle = "idle"
    backend_planning = "backend_planning"
    backend_execution = "backend_execution"
    backend_validation = "backend_validation"
    frontend_planning = "frontend_planning"
    frontend_execution = "frontend_execution"
    frontend_validation = "frontend_validation"
    integration_check = "integration_check"


class PlanStep(BaseModel):
    """A single step in an implementation plan."""

    step_id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "skipped"] = "pending"
    completed_at: datetime | None = None

    def mark_completed(self) -> PlanStep:
        """Return a copy marked as completed with the current timestamp."""
        return self.model_copy(
            update={
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
            }
        )

    def mark_in_progress(self) -> PlanStep:
        """Return a copy marked as in-progress."""
        return self.model_copy(update={"status": "in_progress"})


class CheckpointState(BaseModel):
    """A named checkpoint within an execution session."""

    name: str
    description: str = ""
    phase: ExecutionPhase = ExecutionPhase.idle
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionState(BaseModel):
    """Complete persistent orchestration state for a session.

    This is persisted as JSON and injected into provider requests so
    that any model — regardless of provider — can seamlessly continue
    execution from the last known state.
    """

    session_id: str
    active_model: str = ""
    current_checkpoint: CheckpointState | None = None
    implementation_phase: ExecutionPhase = ExecutionPhase.idle
    approved_plan: str | None = None
    completed_steps: list[PlanStep] = Field(default_factory=list)
    remaining_steps: list[PlanStep] = Field(default_factory=list)
    locked_rules: list[str] = Field(default_factory=list)
    active_files: list[str] = Field(default_factory=list)
    architecture_sources: list[str] = Field(default_factory=list)
    validation_findings: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def touch(self) -> None:
        """Update the ``last_updated`` timestamp in place."""
        self.last_updated = datetime.now(timezone.utc)

    def progress_summary(self) -> tuple[int, int]:
        """Return ``(completed_count, total_count)``."""
        completed = sum(1 for s in self.completed_steps if s.status == "completed")
        total = len(self.completed_steps) + len(self.remaining_steps)
        return completed, total

    def has_active_plan(self) -> bool:
        """Return whether an approved plan with steps exists."""
        return self.approved_plan is not None and (
            bool(self.completed_steps) or bool(self.remaining_steps)
        )


class ExecutionStateUpdate(BaseModel):
    """Partial update payload accepted by the REST API.

    Only non-None fields are applied to the existing state.
    """

    active_model: str | None = None
    current_checkpoint: CheckpointState | None = None
    implementation_phase: ExecutionPhase | None = None
    approved_plan: str | None = None
    completed_steps: list[PlanStep] | None = None
    remaining_steps: list[PlanStep] | None = None
    locked_rules: list[str] | None = None
    active_files: list[str] | None = None
    architecture_sources: list[str] | None = None
    validation_findings: list[str] | None = None
