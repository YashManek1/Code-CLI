"""Build and inject hidden orchestration context into provider requests.

The injector prepends a compact, token-efficient ``<execution_state>``
block to the system message of every outbound request.  This block
carries the full orchestration state so that any model — regardless
of provider — can seamlessly continue execution.

Design choices:
- YAML-like compact format (~200-400 tokens typical, capped at ~800).
- Idempotent: existing execution_state blocks are replaced, never duplicated.
- Provider-safe: works with both OpenAI ``messages[0].content`` and
  Anthropic ``system`` field injection.
- Deterministic: identical state always produces identical output.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..models.execution_state import ExecutionState

_BLOCK_OPEN = "<execution_state>"
_BLOCK_CLOSE = "</execution_state>"
_MAX_PLAN_CHARS = 600  # Truncate approved plan beyond this


def build_orchestration_context(state: ExecutionState) -> str:
    """Build the token-efficient orchestration context block.

    Returns an empty string when there is nothing meaningful to inject
    (idle phase with no plan and no checkpoint).
    """
    if (
        state.implementation_phase == "idle"
        and state.current_checkpoint is None
        and not state.approved_plan
        and not state.completed_steps
        and not state.remaining_steps
    ):
        return ""

    lines: list[str] = [_BLOCK_OPEN]

    # Checkpoint
    if state.current_checkpoint:
        lines.append(f"CHECKPOINT: {state.current_checkpoint.name}")

    # Phase
    lines.append(f"PHASE: {state.implementation_phase}")

    # Plan status
    if state.approved_plan:
        lines.append("PLAN_STATUS: locked")
        # Include a truncated plan summary
        plan_preview = state.approved_plan[:_MAX_PLAN_CHARS]
        if len(state.approved_plan) > _MAX_PLAN_CHARS:
            plan_preview += "... [truncated]"
        lines.append(f"PLAN_SUMMARY: {plan_preview}")

    # Completed steps
    if state.completed_steps:
        lines.append("COMPLETED:")
        for step in state.completed_steps:
            lines.append(f"- [x] {step.description}")

    # Remaining steps
    if state.remaining_steps:
        lines.append("REMAINING:")
        for step in state.remaining_steps:
            marker = "/" if step.status == "in_progress" else " "
            lines.append(f"- [{marker}] {step.description}")

    # Locked rules
    if state.locked_rules:
        lines.append("RULES:")
        for rule in state.locked_rules:
            lines.append(f"- {rule}")

    # Active files
    if state.active_files:
        lines.append("ACTIVE_FILES:")
        for filepath in state.active_files[:20]:  # Cap at 20 files
            lines.append(f"- {filepath}")

    # Validation findings
    if state.validation_findings:
        lines.append("VALIDATION:")
        for finding in state.validation_findings[:10]:  # Cap at 10
            lines.append(f"- {finding}")

    lines.append(_BLOCK_CLOSE)
    return "\n".join(lines)


def _strip_existing_block(text: str) -> str:
    """Remove any existing ``<execution_state>`` block from text."""
    start = text.find(_BLOCK_OPEN)
    if start == -1:
        return text
    end = text.find(_BLOCK_CLOSE, start)
    if end == -1:
        return text
    end += len(_BLOCK_CLOSE)
    # Remove the block and any surrounding whitespace
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and after:
        return before + "\n\n" + after
    return before + after


def inject_execution_state_context(
    body: dict[str, Any],
    state: ExecutionState | None,
) -> dict[str, Any]:
    """Inject orchestration state into an OpenAI-format request body.

    The state block is prepended as a system message at the start of
    the ``messages`` array.  If a system message already exists at
    position 0, the block is prepended to its content.
    """
    if state is None:
        return body

    context = build_orchestration_context(state)
    if not context:
        return body

    messages = body.get("messages", [])
    if not messages:
        return body

    # Check if first message is a system message
    first = messages[0]
    if isinstance(first, dict) and first.get("role") == "system":
        existing_content = first.get("content", "")
        cleaned = _strip_existing_block(existing_content)
        new_content = context + "\n\n" + cleaned if cleaned else context
        messages[0] = {**first, "content": new_content}
    else:
        # Prepend a new system message
        system_msg = {"role": "system", "content": context}
        messages.insert(0, system_msg)

    body["messages"] = messages

    logger.debug(
        "STATE_INJECT_OPENAI: session={} phase={} context_len={}",
        state.session_id,
        state.implementation_phase,
        len(context),
    )
    return body


def inject_execution_state_context_anthropic(
    body: dict[str, Any],
    state: ExecutionState | None,
) -> dict[str, Any]:
    """Inject orchestration state into an Anthropic-format request body.

    The Anthropic Messages API uses a top-level ``system`` field which
    can be a string or a list of ``{"type": "text", "text": ...}``
    content blocks.  The state block is prepended to the existing
    system content.
    """
    if state is None:
        return body

    context = build_orchestration_context(state)
    if not context:
        return body

    existing_system = body.get("system")

    if existing_system is None:
        body["system"] = context
    elif isinstance(existing_system, str):
        cleaned = _strip_existing_block(existing_system)
        body["system"] = context + "\n\n" + cleaned if cleaned else context
    elif isinstance(existing_system, list):
        # List of SystemContent blocks — prepend as a new text block
        # First, remove any existing execution_state block
        filtered_blocks: list[dict[str, Any]] = []
        for block in existing_system:
            if isinstance(block, dict) and block.get("type") == "text":
                cleaned_text = _strip_existing_block(block.get("text", ""))
                if cleaned_text.strip():
                    filtered_blocks.append({**block, "text": cleaned_text})
            else:
                filtered_blocks.append(block)

        state_block = {"type": "text", "text": context}
        body["system"] = [state_block] + filtered_blocks

    logger.debug(
        "STATE_INJECT_ANTHROPIC: session={} phase={} context_len={}",
        state.session_id,
        state.implementation_phase,
        len(context),
    )
    return body
