"""Optimization handlers for fast-path API responses.

Each handler returns a MessagesResponse if the request matches and the
optimization is enabled, otherwise None.

MiniMax / weak-model additions
-------------------------------
- ``try_minimax_tool_guard`` - intercepts requests whose tool-call content
  would produce argument JSON larger than the model's ``max_tool_tokens`` cap
  *before* the request leaves the proxy.  This prevents upstream timeouts and
  silent truncation when MiniMax tries to emit a 50 KB Terraform file in a
  single Write tool call.
- ``try_large_write_split_hint`` - detects oversized ``content`` fields in
  FileWrite / Write tool results and injects an assistant text nudge asking
  the model to split the operation.  Keeps Claude Code's retry loop clean.
"""

from __future__ import annotations

import uuid

from loguru import logger

from config.settings import Settings

from .command_utils import extract_command_prefix, extract_filepaths_from_command
from .detection import (
    is_filepath_extraction_request,
    is_prefix_detection_request,
    is_quota_check_request,
    is_suggestion_mode_request,
    is_title_generation_request,
)
from .models.anthropic import MessagesRequest
from .models.responses import MessagesResponse, Usage


# Lazy import - avoids circular dep at module init time.
def _get_quirks(model: str):  # type: ignore[return]
    try:
        from core.anthropic.tools import get_model_quirks

        return get_model_quirks(model)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _text_response(
    request_data: MessagesRequest,
    text: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> MessagesResponse:
    return MessagesResponse(
        id=f"msg_{uuid.uuid4()}",
        model=request_data.model,
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# Original optimizations (unchanged logic, kept for back-compat)
# ---------------------------------------------------------------------------


def try_prefix_detection(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Fast prefix detection - return command prefix without API call."""
    if not settings.fast_prefix_detection:
        return None
    is_prefix_req, command = is_prefix_detection_request(request_data)
    if not is_prefix_req:
        return None
    logger.info("Optimization: Fast prefix detection request")
    return _text_response(
        request_data,
        extract_command_prefix(command),
        input_tokens=100,
        output_tokens=5,
    )


def try_quota_mock(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Mock quota probe requests."""
    if not settings.enable_network_probe_mock:
        return None
    if not is_quota_check_request(request_data):
        return None
    logger.info("Optimization: Intercepted and mocked quota probe")
    return _text_response(
        request_data,
        "Quota check passed.",
        input_tokens=10,
        output_tokens=5,
    )


def try_title_skip(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Skip title generation requests."""
    if not settings.enable_title_generation_skip:
        return None
    if not is_title_generation_request(request_data):
        return None
    logger.info("Optimization: Skipped title generation request")
    return _text_response(
        request_data,
        "Conversation",
        input_tokens=100,
        output_tokens=5,
    )


def try_suggestion_skip(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Skip suggestion mode requests."""
    if not settings.enable_suggestion_mode_skip:
        return None
    if not is_suggestion_mode_request(request_data):
        return None
    logger.info("Optimization: Skipped suggestion mode request")
    return _text_response(
        request_data,
        "",
        input_tokens=100,
        output_tokens=1,
    )


def try_filepath_mock(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Mock filepath extraction requests."""
    if not settings.enable_filepath_extraction_mock:
        return None
    is_filepath_request, command, command_output = is_filepath_extraction_request(
        request_data
    )
    if not is_filepath_request:
        return None
    filepaths = extract_filepaths_from_command(command, command_output)
    logger.info("Optimization: Mocked filepath extraction")
    return _text_response(
        request_data,
        filepaths,
        input_tokens=100,
        output_tokens=10,
    )


# ---------------------------------------------------------------------------
# New: MiniMax / weak-model guards
# ---------------------------------------------------------------------------


def _extract_tool_result_content(messages: list) -> list[tuple[str, str, int]]:
    """Return list of (tool_name, content, char_len) for tool_result messages.

    Inspects the last user message for tool_result blocks so we can detect
    oversized content that would cause the model to repeat a giant write.
    """
    results: list[tuple[str, str, int]] = []
    if not messages:
        return results
    last_message = messages[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user":
        return results
    content = last_message.get("content", [])
    if not isinstance(content, list):
        return results
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        tool_result_content = block.get("content", "")
        if isinstance(tool_result_content, list):
            tool_result_content = " ".join(
                part.get("text", "")
                for part in tool_result_content
                if isinstance(part, dict)
            )
        tool_result_text = str(tool_result_content)
        results.append(
            (block.get("tool_use_id", "?"), tool_result_text, len(tool_result_text))
        )
    return results


def try_minimax_tool_guard(
    request_data: MessagesRequest, _settings: Settings
) -> MessagesResponse | None:
    """Reject requests where any pending tool result exceeds the model's cap.

    When Claude Code feeds back a massive FileWrite result (e.g. a 40 KB
    Terraform plan), MiniMax will try to re-emit it all in one chunk and
    produce truncated / broken JSON.  We intercept here and return a text
    response telling the model to split the operation.
    """
    quirks = _get_quirks(request_data.model)
    if quirks is None or not quirks.max_tool_tokens:
        return None

    cap = quirks.max_tool_tokens
    messages = getattr(request_data, "messages", []) or []

    # Convert Pydantic models / dicts uniformly
    raw_messages: list[dict] = []
    for message in messages:
        if isinstance(message, dict):
            raw_messages.append(message)
        elif hasattr(message, "model_dump"):
            raw_messages.append(message.model_dump())
        else:
            raw_messages.append(
                {
                    "role": getattr(message, "role", ""),
                    "content": getattr(message, "content", ""),
                }
            )

    for tool_id, _content, char_len in _extract_tool_result_content(raw_messages):
        if char_len > cap:
            logger.warning(
                "TOOL_GUARD: tool_result id={} len={} > cap={} for model '{}'; "
                "returning split-hint response",
                tool_id,
                char_len,
                cap,
                request_data.model,
            )
            hint = (
                f"The previous tool result was {char_len:,} characters which exceeds "
                f"the {cap:,}-character limit for this model. "
                "Please split the operation into smaller chunks (e.g. write one "
                "file section at a time, or use shorter content blocks) and retry."
            )
            return _text_response(
                request_data,
                hint,
                input_tokens=50,
                output_tokens=60,
            )

    return None


def try_large_write_split_hint(
    request_data: MessagesRequest, _settings: Settings
) -> MessagesResponse | None:
    """Detect when the last assistant message requested an oversized Write/Edit.

    If the second-to-last message was an assistant tool_use with a ``content``
    or ``new_content`` field larger than the model cap, we nudge the model to
    split before it tries again.
    """
    quirks = _get_quirks(request_data.model)
    if quirks is None or not quirks.max_tool_tokens:
        return None

    cap = quirks.max_tool_tokens
    messages = getattr(request_data, "messages", []) or []

    # Scan last 5 messages for an oversized assistant tool_use.
    if len(messages) < 2:
        return None

    for previous_message in reversed(messages[-5:]):
        if isinstance(previous_message, dict):
            role = previous_message.get("role", "")
            content = previous_message.get("content", [])
        elif hasattr(previous_message, "role"):
            role = previous_message.role
            content = getattr(previous_message, "content", [])
        else:
            continue

        if role != "assistant":
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                block_input = block.get("input", {})
            elif hasattr(block, "type"):
                block_type = block.type
                block_input = getattr(block, "input", {}) or {}
            else:
                continue

            if block_type != "tool_use":
                continue
            if not isinstance(block_input, dict):
                continue

            for field_name in ("content", "new_content", "new_string", "text"):
                field_value = block_input.get(field_name, "")
                if isinstance(field_value, str) and len(field_value) > cap:
                    tool_name = (
                        block.get("name", "tool")
                        if isinstance(block, dict)
                        else getattr(block, "name", "tool")
                    )
                    logger.warning(
                        "WRITE_HINT: tool='{}' field='{}' len={} > cap={}; injecting split hint",
                        tool_name,
                        field_name,
                        len(field_value),
                        cap,
                    )
                    hint = (
                        f"The {tool_name} call attempted to write {len(field_value):,} characters "
                        f"in a single operation (cap: {cap:,}). "
                        "Please split the content across multiple smaller write operations."
                    )
                    return _text_response(
                        request_data, hint, input_tokens=50, output_tokens=60
                    )

    return None


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

# Cheapest / most-specific optimizations first for faster short-circuit.
OPTIMIZATION_HANDLERS = [
    try_quota_mock,
    try_prefix_detection,
    try_minimax_tool_guard,  # NEW - intercept oversized tool results early
    try_large_write_split_hint,  # NEW - prevent giant re-writes
    try_title_skip,
    try_suggestion_skip,
    try_filepath_mock,
]


def try_optimizations(
    request_data: MessagesRequest, settings: Settings
) -> MessagesResponse | None:
    """Run optimization handlers in order. Returns first match or None."""
    for handler in OPTIMIZATION_HANDLERS:
        result = handler(request_data, settings)
        if result is not None:
            return result
    return None
