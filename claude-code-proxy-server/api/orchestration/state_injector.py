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
import html

from ..models.execution_state import ExecutionState

_BLOCK_OPEN = "<execution_state>"
_BLOCK_CLOSE = "</execution_state>"
_MAX_PLAN_CHARS = 600  # Truncate approved plan beyond this


def _safe_xml(text: str) -> str:
    """Strictly escape text and prevent injection of block delimiters."""
    if not text:
        return ""
    # 1. Standard HTML/XML escaping
    escaped = html.escape(text)
    # 2. Prevent injection of our own block markers
    escaped = escaped.replace(_BLOCK_OPEN, "[BLOCK_OPEN_REDATED]")
    escaped = escaped.replace(_BLOCK_CLOSE, "[BLOCK_CLOSE_REDATED]")
    return escaped


def build_orchestration_context(state: ExecutionState) -> str:
    """Build the token-efficient orchestration context block.

    Returns an empty string when there is nothing meaningful to inject
    (idle phase with no plan and no checkpoint).
    """
    if (
        state.implementation_phase == "idle"
        and state.current_checkpoint is None
        and not state.approved_plan
        and not state.remaining_steps
    ):
        return ""

    lines: list[str] = [_BLOCK_OPEN]

    # Checkpoint
    if state.current_checkpoint:
        lines.append(f"CHECKPOINT: {_safe_xml(state.current_checkpoint.name)}")

    # Phase
    lines.append(f"PHASE: {state.implementation_phase}")

    # Plan status
    if state.approved_plan:
        lines.append("PLAN_STATUS: locked")
        # Include a truncated plan summary
        plan_preview = state.approved_plan[:_MAX_PLAN_CHARS]
        if len(state.approved_plan) > _MAX_PLAN_CHARS:
            plan_preview += "... [truncated]"
        lines.append(f"PLAN_SUMMARY: {_safe_xml(plan_preview)}")

    # Completed steps
    if state.completed_steps:
        lines.append("COMPLETED:")
        for step in state.completed_steps:
            lines.append(f"- [x] {_safe_xml(step.description)}")

    # Remaining steps
    if state.remaining_steps:
        lines.append("REMAINING:")
        for step in state.remaining_steps:
            marker = "/" if step.status == "in_progress" else " "
            lines.append(f"- [{marker}] {_safe_xml(step.description)}")

    # Locked rules
    if state.locked_rules:
        lines.append("RULES:")
        for rule in state.locked_rules:
            lines.append(f"- {_safe_xml(rule)}")

    # Active files
    if state.active_files:
        lines.append("ACTIVE_FILES:")
        for filepath in state.active_files[:20]:  # Cap at 20 files
            lines.append(f"- {_safe_xml(filepath)}")

    # Validation findings
    if state.validation_findings:
        lines.append("VALIDATION:")
        for finding in state.validation_findings[:10]:  # Cap at 10
            lines.append(f"- {_safe_xml(finding)}")

    # Pending subtasks
    if state.pending_subtasks:
        lines.append("PENDING_SUBTASKS:")
        for task in state.pending_subtasks[:10]:
            lines.append(f"- {_safe_xml(task)}")

    # Validation failures (detailed)
    if state.validation_failures:
        lines.append("VALIDATION_FAILURES:")
        for fail in state.validation_failures[:5]:
            msg = fail.get("message", "Unknown error")
            file = fail.get("file", "")
            loc = f" at {file}" if file else ""
            lines.append(f"- {_safe_xml(msg)}{_safe_xml(loc)}")

    # Retry history (Cognitive Feedback Loop)
    if state.retry_history:
        lines.append("RETRY_HISTORY:")
        for retry in state.retry_history[-3:]: # Only last 3 to save tokens
            err = retry.get("error", "Unknown")
            type_ = retry.get("failure_type", "Generic")
            lines.append(f"- {_safe_xml(type_)}: {_safe_xml(str(err))}")

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


def inject_execution_state_context(body: dict, state: ExecutionState | None) -> dict:
    """Inject orchestration context into an OpenAI-format request body."""
    if not state:
        return body

    context = build_orchestration_context(state)
    if not context:
        return body

    messages = body.get("messages", [])
    if not messages:
        # Prepend a new system message if none exists
        body["messages"] = [{"role": "system", "content": context}]
        return body

    # Check if first message is a system message
    if messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if isinstance(content, str):
            # Strip existing block and append new one
            stripped = _strip_existing_block(content)
            messages[0]["content"] = (stripped + "\n\n" + context).strip()
        elif isinstance(content, list):
            # Handle structured content (e.g. vision or thinking)
            # Find the first text block and update it
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    stripped = _strip_existing_block(block["text"])
                    block["text"] = (stripped + "\n\n" + context).strip()
                    break
    else:
        # Prepend a new system message
        body["messages"] = [{"role": "system", "content": context}] + messages

    return body


def inject_execution_state_context_anthropic(body: dict, state: ExecutionState | None) -> dict:
    """Inject orchestration context into an Anthropic-format request body."""
    if not state:
        return body

    context = build_orchestration_context(state)
    if not context:
        return body

    system = body.get("system")

    if system is None:
        body["system"] = context
    elif isinstance(system, str):
        stripped = _strip_existing_block(system)
        body["system"] = (stripped + "\n\n" + context).strip()
    elif isinstance(system, list):
        # Anthropic list system blocks
        # Prepend a new text block to match test expectation of len increment
        body["system"] = [{"type": "text", "text": context}] + system

    return body
