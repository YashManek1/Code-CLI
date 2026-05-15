"""Native Anthropic Messages request builder for OpenRouter."""

from __future__ import annotations

from typing import Any

from loguru import logger
from config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS as OPENROUTER_DEFAULT_MAX_TOKENS,
)
from core.anthropic.native_messages_request import (
    OpenRouterExtraBodyError,
    build_openrouter_native_request_body,
)
from providers.exceptions import InvalidRequestError

_OPENROUTER_DEEPSEEK_R1_MAX_TOKENS = 16000
_OPENROUTER_DEEPSEEK_R1_0528_MAX_TOKENS = 163840


def _is_deepseek_r1_free_model(model: str) -> bool:
    return model.startswith("deepseek/deepseek-r1") and ":free" in model


def _apply_deepseek_r1_free_compatibility(body: dict[str, Any]) -> None:
    """Sanitize fields commonly rejected by OpenRouter free DeepSeek R1 routes."""
    if body.pop("mcp_servers", None) is not None:
        logger.warning(
            "OPENROUTER_REQUEST: removed mcp_servers for deepseek-r1 free compatibility"
        )

    if body.pop("reasoning", None) is not None:
        logger.warning(
            "OPENROUTER_REQUEST: removed reasoning for deepseek-r1 free compatibility"
        )

    had_tools = bool(body.get("tools"))
    body.pop("tools", None)
    body.pop("tool_choice", None)
    if had_tools:
        logger.warning(
            "OPENROUTER_REQUEST: removed tools/tool_choice for deepseek-r1 free compatibility"
        )


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build an Anthropic-format request body for OpenRouter's messages API."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    try:
        body = build_openrouter_native_request_body(
            request_data,
            thinking_enabled=thinking_enabled,
            default_max_tokens=OPENROUTER_DEFAULT_MAX_TOKENS,
        )
    except OpenRouterExtraBodyError as exc:
        raise InvalidRequestError(str(exc)) from exc

    model = body.get("model")
    if isinstance(model, str):
        if _is_deepseek_r1_free_model(model):
            _apply_deepseek_r1_free_compatibility(body)

        max_tokens = body.get("max_tokens")
        if model.startswith("deepseek/deepseek-r1-0528"):
            cap = _OPENROUTER_DEEPSEEK_R1_0528_MAX_TOKENS
        elif model.startswith("deepseek/deepseek-r1"):
            cap = _OPENROUTER_DEEPSEEK_R1_MAX_TOKENS
        else:
            cap = None

        if isinstance(max_tokens, int) and cap is not None and max_tokens > cap:
            body["max_tokens"] = cap

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
