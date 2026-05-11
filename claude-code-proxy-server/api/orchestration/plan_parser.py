"""Parse approved plans into structured PlanStep objects.

Handles common markdown plan formats:
- ``- [ ] step description``
- ``- [x] completed step``
- ``1. step description``
- ``### Phase N — Title`` (section headers)

Step IDs are generated deterministically from description content.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from ..models.execution_state import PlanStep

# Patterns for plan step extraction
_CHECKBOX_PATTERN = re.compile(
    r"^\s*[-*]\s*\[([ xX/])\]\s*(.+)$", re.MULTILINE
)
_NUMBERED_PATTERN = re.compile(
    r"^\s*(\d+)[.)]\s+(.+)$", re.MULTILINE
)
_BULLET_PATTERN = re.compile(
    r"^\s*[-*]\s+(.+)$",
    re.MULTILINE,
)


def _generate_step_id(description: str) -> str:
    """Generate a deterministic step ID from a description."""
    normalized = description.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _status_from_checkbox(
    marker: str,
) -> Literal["pending", "in_progress", "completed", "skipped"]:
    """Map a checkbox marker to a PlanStep status."""
    marker = marker.strip().lower()
    if marker == "x":
        return "completed"
    if marker == "/":
        return "in_progress"
    return "pending"


def parse_plan_text(plan_markdown: str) -> list[PlanStep]:
    """Parse a markdown plan into a list of PlanStep objects.

    Recognises checkbox lists (``- [ ] ...``) and numbered lists.
    Checkbox markers determine initial status: ``[x]`` → completed,
    ``[/]`` → in_progress, ``[ ]`` → pending.
    """
    steps: list[PlanStep] = []
    seen_ids: set[str] = set()

    # First pass: checkbox items
    for match in _CHECKBOX_PATTERN.finditer(plan_markdown):
        marker, description = match.group(1), match.group(2).strip()
        if not description:
            continue
        step_id = _generate_step_id(description)
        if step_id in seen_ids:
            continue
        seen_ids.add(step_id)
        steps.append(
            PlanStep(
                step_id=step_id,
                description=description,
                status=_status_from_checkbox(marker),
            )
        )

    # Second pass: numbered items
    if not steps:
        for match in _NUMBERED_PATTERN.finditer(plan_markdown):
            description = match.group(2).strip()
            if not description:
                continue
            step_id = _generate_step_id(description)
            if step_id in seen_ids:
                continue
            seen_ids.add(step_id)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    status="pending",
                )
            )
    # Third pass: plain bullet lists
    if not steps:
        for match in _BULLET_PATTERN.finditer(plan_markdown):
            description = match.group(1).strip()

            if not description:
                continue

            lowered = description.lower()

            # Skip phase headings
            if lowered.startswith("phase "):
                continue

            step_id = _generate_step_id(description)

            if step_id in seen_ids:
                continue

            seen_ids.add(step_id)

            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    status="pending",
                )
            )

    return steps


def normalize_plan(steps: list[PlanStep]) -> list[PlanStep]:
    """De-duplicate steps and ensure each has a unique step_id."""
    seen: set[str] = set()
    unique: list[PlanStep] = []
    for step in steps:
        if step.step_id in seen:
            continue
        seen.add(step.step_id)
        unique.append(step)
    return unique


def split_by_status(
    steps: list[PlanStep],
) -> tuple[list[PlanStep], list[PlanStep]]:
    """Split steps into (completed, remaining) lists."""
    completed: list[PlanStep] = []
    remaining: list[PlanStep] = []
    for step in steps:
        if step.status == "completed":
            completed.append(step)
        else:
            remaining.append(step)
    return completed, remaining
