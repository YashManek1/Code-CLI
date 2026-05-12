"""Application services for the Claude-compatible API."""

from __future__ import annotations

import json
import re
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS, SSEBuilder
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .execution_state_store import ExecutionStateStore
from .model_router import ModelRouter
from .models.anthropic import (
    MessagesRequest,
    SystemContent,
    ThinkingConfig,
    TokenCountRequest,
)
from .models.responses import MessagesResponse, TokenCountResponse
from .optimization_handlers import try_optimizations
from .orchestration.state_injector import build_orchestration_context
from .response_cache import dedupe_and_cache_stream
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim"})
_PLAN_ONLY_AGENT_TOOL_NAMES = frozenset({"Agent", "Task"})
_PLAN_ONLY_PHRASES = (
    "stop after planning",
    "stop after plan",
    "only plan",
    "plan only",
    "do not implement",
    "don't implement",
)
_AGENT_RETRY_ERROR_MARKERS = (
    "agent type",
    "not found",
    "available agents",
)
_SIMPLE_PROMPT_MAX_CHARS = 260
_SIMPLE_PROMPT_PREFIXES = (
    "what is ",
    "what are ",
    "who is ",
    "define ",
    "explain ",
    "summarize ",
    "rewrite ",
    "translate ",
    "answer ",
)
_COMPLEX_PROMPT_WORDS = (
    "plan",
    "implement",
    "refactor",
    "debug",
    "fix",
    "solve",
    "issue",
    "test",
    "analyze the code",
    "bottleneck",
    "performance",
    "increase speed",
    "review",
)
_WEAK_MODEL_HINT = (
    "Provider quality guard: answer directly and avoid repeating yourself. "
    "Never print serialized content-block wrappers such as [{'type':'text',...}]. "
    "Use valid tool calls only when needed, and do not launch subagents unless "
    "the user explicitly asks for delegation or parallel agents."
)


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def _field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _field(message, "role") != "user":
            continue
        content = _field(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = _field(block, "text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return ""
    return ""


def _iter_recent_user_text(messages: list[Any], *, max_messages: int = 6) -> list[str]:
    texts: list[str] = []
    for message in reversed(messages[-max_messages:]):
        if _field(message, "role") != "user":
            continue
        content = _field(message, "content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = _field(block, "type")
            if block_type == "tool_result":
                value = _field(block, "content", "")
            else:
                value = _field(block, "text", "")
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.extend(
                    str(_field(item, "text", ""))
                    for item in value
                    if _field(item, "text", "")
                )
    return texts


def _is_plan_only_request(request_data: MessagesRequest) -> bool:
    text = _latest_user_text(request_data.messages).lower()
    return any(phrase in text for phrase in _PLAN_ONLY_PHRASES)


def _disable_subagents_for_plan_only_request(request_data: MessagesRequest) -> None:
    """Remove subagent tools when the user explicitly asks to stop at planning."""
    if not request_data.tools or not _is_plan_only_request(request_data):
        return

    original_count = len(request_data.tools)
    request_data.tools = [
        tool
        for tool in request_data.tools
        if getattr(tool, "name", "") not in _PLAN_ONLY_AGENT_TOOL_NAMES
    ]
    if not request_data.tools:
        request_data.tools = None

    tool_choice_name = (
        request_data.tool_choice.get("name")
        if isinstance(request_data.tool_choice, dict)
        else None
    )
    if tool_choice_name in _PLAN_ONLY_AGENT_TOOL_NAMES:
        request_data.tool_choice = None

    removed_count = original_count - len(request_data.tools or [])
    if removed_count:
        logger.info(
            "PLAN_ONLY_TOOL_GUARD: removed_subagent_tools={} latest_user_request=true",
            removed_count,
        )


def _disable_subagents_after_agent_error(request_data: MessagesRequest) -> None:
    """Avoid wasting another model/tool turn after an invalid agent selection."""
    if not request_data.tools:
        return

    for text in _iter_recent_user_text(request_data.messages):
        lowered = text.lower()
        if all(marker in lowered for marker in _AGENT_RETRY_ERROR_MARKERS):
            original_count = len(request_data.tools)
            request_data.tools = [
                tool
                for tool in request_data.tools
                if getattr(tool, "name", "") not in _PLAN_ONLY_AGENT_TOOL_NAMES
            ]
            if not request_data.tools:
                request_data.tools = None
            removed_count = original_count - len(request_data.tools or [])
            if removed_count:
                logger.info(
                    "AGENT_RETRY_GUARD: removed_subagent_tools={} after_agent_error=true",
                    removed_count,
                )
            return


def _maybe_disable_thinking_for_simple_prompt(
    request_data: MessagesRequest, settings: Settings
) -> None:
    """Trade hidden reasoning for faster visible output on simple no-tool turns."""
    if not settings.auto_disable_thinking_simple_prompts:
        return
    if request_data.tools or request_data.thinking is not None:
        return

    text = _latest_user_text(request_data.messages).strip()
    lowered = text.lower()
    if not text or len(text) > _SIMPLE_PROMPT_MAX_CHARS:
        return
    if "```" in text or any(
        re.search(rf"\b{re.escape(word)}\b", lowered)
        for word in _COMPLEX_PROMPT_WORDS
    ):
        return
    if lowered.endswith("?") or lowered.startswith(_SIMPLE_PROMPT_PREFIXES):
        request_data.thinking = ThinkingConfig(type="disabled", enabled=False)
        logger.debug(
            "FAST_THINKING_GUARD: disabled thinking for simple no-tool prompt len={}",
            len(text),
        )


def _prepend_system_text(request_data: MessagesRequest, text: str) -> None:
    if request_data.system is None:
        request_data.system = text
    elif isinstance(request_data.system, str):
        if text not in request_data.system:
            request_data.system = f"{text}\n\n{request_data.system}"
    elif isinstance(request_data.system, list) and not any(
        block.text == text for block in request_data.system
    ):
        request_data.system = [
            SystemContent(type="text", text=text),
            *request_data.system,
        ]


def _inject_weak_model_quality_hint(
    request_data: MessagesRequest, settings: Settings
) -> None:
    if not settings.enable_weak_model_quality_hints:
        return
    try:
        from core.anthropic.tools import get_model_quirks
    except ImportError:
        return

    quirks = get_model_quirks(request_data.model)
    if not (quirks.requires_json_repair or quirks.flatten_tool_schemas):
        return
    _prepend_system_text(request_data, _WEAK_MODEL_HINT)


def _messages_response_from_sse_text(text: str) -> MessagesResponse:
    from core.anthropic.stream_contracts import parse_sse_text

    events = parse_sse_text(text)
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}

    for event in events:
        data = event.data
        if event.event == "message_start":
            raw_message = data.get("message", {})
            if isinstance(raw_message, dict):
                message = raw_message
                raw_usage = raw_message.get("usage")
                if isinstance(raw_usage, dict):
                    usage.update(raw_usage)
            continue

        if event.event == "content_block_start":
            index = data.get("index")
            block = data.get("content_block", {})
            if isinstance(index, int) and isinstance(block, dict):
                blocks[index] = dict(block)
            continue

        if event.event == "content_block_delta":
            index = data.get("index")
            delta = data.get("delta", {})
            if not isinstance(index, int) or not isinstance(delta, dict):
                continue
            block = blocks.setdefault(index, {"type": "text", "text": ""})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
            elif delta_type == "thinking_delta":
                block["thinking"] = str(block.get("thinking", "")) + str(
                    delta.get("thinking", "")
                )
            elif delta_type == "input_json_delta":
                block["_partial_json"] = str(block.get("_partial_json", "")) + str(
                    delta.get("partial_json", "")
                )
            continue

        if event.event == "content_block_stop":
            index = data.get("index")
            if not isinstance(index, int) or index not in blocks:
                continue
            block = blocks.pop(index)
            if block.get("type") == "tool_use":
                raw_input = str(block.pop("_partial_json", ""))
                if raw_input:
                    try:
                        block["input"] = json.loads(raw_input)
                    except json.JSONDecodeError:
                        block["input"] = {}
            content.append(block)
            continue

        if event.event == "message_delta":
            delta = data.get("delta", {})
            if isinstance(delta, dict):
                stop_reason = str(delta.get("stop_reason") or stop_reason)
            raw_usage = data.get("usage")
            if isinstance(raw_usage, dict):
                usage.update(raw_usage)

    content.extend(blocks.values())

    return MessagesResponse(
        id=str(message.get("id") or f"msg_{uuid.uuid4()}"),
        model=str(message.get("model") or "unknown"),
        content=content,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage={
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        },
    )


async def _stream_messages_response(
    response: MessagesResponse,
    *,
    log_raw_events: bool = False,
) -> AsyncIterator[str]:
    """Convert a complete Anthropic message response into a valid SSE stream."""
    sse = SSEBuilder(
        response.id,
        response.model,
        response.usage.input_tokens,
        log_raw_events=log_raw_events,
    )
    yield sse.message_start()

    for block in response.content:
        block_type = _field(block, "type")
        if block_type == "text":
            text = str(_field(block, "text", ""))
            if text:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(text)
            continue

        if block_type == "thinking":
            thinking = str(_field(block, "thinking", ""))
            if thinking:
                for event in sse.ensure_thinking_block():
                    yield event
                yield sse.emit_thinking_delta(thinking)
            continue

        if block_type == "tool_use":
            for event in sse.close_content_blocks():
                yield event
            tool_index = len(sse.blocks.tool_states)
            tool_id = str(_field(block, "id", f"toolu_{uuid.uuid4().hex[:12]}"))
            name = str(_field(block, "name", "tool"))
            input_obj = _field(block, "input", {}) or {}
            yield sse.start_tool_block(tool_index, tool_id, name)
            delta = sse.emit_tool_delta(
                tool_index,
                json.dumps(input_obj, ensure_ascii=False, separators=(",", ":")),
            )
            if delta is not None:
                yield delta
            yield sse.stop_tool_block(tool_index)

    for event in sse.close_content_blocks():
        yield event
    yield sse.message_delta(response.stop_reason or "end_turn", response.usage.output_tokens)
    yield sse.message_stop()


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        execution_state_store: ExecutionStateStore | None = None,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._execution_state_store = execution_state_store

    # ------------------------------------------------------------------
    # Execution-state orchestration helpers
    # ------------------------------------------------------------------

    def _extract_session_ids(
        self, request_data: MessagesRequest
    ) -> tuple[str | None, str | None]:
        """Extract current and parent session ids from headers or metadata."""
        session_id: str | None = None
        parent_session_id: str | None = None

        # Check forwarded headers
        if request_data.forwarded_headers:
            for header_name in ("x-session-id", "anthropic-conversation-id"):
                value = request_data.forwarded_headers.get(header_name)
                if value:
                    session_id = value
                    break
            parent_value = request_data.forwarded_headers.get("x-parent-session-id")
            if parent_value:
                parent_session_id = parent_value

        # Check metadata
        if request_data.metadata:
            direct_session_id = request_data.metadata.get("session_id")
            if direct_session_id:
                session_id = str(direct_session_id)
            direct_parent_session_id = request_data.metadata.get("parent_session_id")
            if direct_parent_session_id:
                parent_session_id = str(direct_parent_session_id)

            user_id = request_data.metadata.get("user_id")
            if isinstance(user_id, str):
                try:
                    parsed_user_id = json.loads(user_id)
                except json.JSONDecodeError:
                    parsed_user_id = None
                if isinstance(parsed_user_id, dict):
                    nested_session_id = parsed_user_id.get("session_id")
                    if nested_session_id:
                        session_id = str(nested_session_id)
                    nested_parent_session_id = parsed_user_id.get("parent_session_id")
                    if nested_parent_session_id:
                        parent_session_id = str(nested_parent_session_id)

        return session_id, parent_session_id

    def _inject_execution_state(
        self,
        request_data: MessagesRequest,
        session_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Inject execution state context into the request's system field."""
        if self._execution_state_store is None or session_id is None:
            return

        state = self._execution_state_store.ensure_state_from_parent(
            session_id,
            parent_session_id,
        )

        # Update active model if it changed
        model_name = request_data.model
        if model_name and state.active_model != model_name:
            state.active_model = model_name
            self._execution_state_store.save(state)

        context = build_orchestration_context(state)
        if not context:
            return

        # Inject into system field (Anthropic format — works for all providers)
        if request_data.system is None:
            request_data.system = context
        elif isinstance(request_data.system, str):
            request_data.system = context + "\n\n" + request_data.system
        elif isinstance(request_data.system, list):
            state_block = SystemContent(type="text", text=context)
            request_data.system = [state_block, *list(request_data.system)]

        logger.debug(
            "EXECUTION_STATE_INJECT: session={} phase={} context_len={}",
            session_id,
            state.implementation_phase,
            len(context),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            _disable_subagents_for_plan_only_request(request_data)
            _disable_subagents_after_agent_error(request_data)
            _maybe_disable_thinking_for_simple_prompt(request_data, self._settings)

            session_id, parent_session_id = self._extract_session_ids(request_data)
            if self._execution_state_store is not None and session_id is not None:
                self._execution_state_store.ensure_state_from_parent(
                    session_id,
                    parent_session_id,
                )

                # Process tool results before context injection so duplicate
                # client retries do not advance state again or create a fresh
                # upstream request body.
                from .orchestration.response_tracker import ResponseTracker

                tracker = ResponseTracker(self._execution_state_store)
                tracker.process_request_messages(session_id, request_data.messages)

            # Inject execution state context before routing
            self._inject_execution_state(request_data, session_id, parent_session_id)

            routed = self._model_router.resolve_messages_request(request_data)
            _inject_weak_model_quality_hint(routed.request, self._settings)
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info("Optimization: Handling Anthropic web server tool")
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                if routed.request.stream is not False:
                    return anthropic_sse_streaming_response(
                        _stream_messages_response(
                            optimized,
                            log_raw_events=self._settings.log_raw_sse_events,
                        )
                    )
                return optimized
            logger.debug("No optimization matched, routing to provider")

            provider = self._provider_getter(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            logger.info(
                "API_REQUEST: request_id={} model={} messages={}",
                request_id,
                routed.request.model,
                len(routed.request.messages),
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )
            provider_stream = provider.stream_response(
                routed.request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=routed.resolved.thinking_enabled,
            )
            return anthropic_sse_streaming_response(
                dedupe_and_cache_stream(
                    routed.request,
                    provider_id=routed.resolved.provider_id,
                    request_id=request_id,
                    factory=lambda: provider_stream,
                ),
            )

        except ProviderError:
            raise
        except Exception as create_message_error:
            _log_unexpected_service_exception(
                self._settings, create_message_error, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(
                    create_message_error
                ),
                detail=get_user_facing_error_message(create_message_error),
            ) from create_message_error

    async def create_message_nonstreaming(
        self, request_data: MessagesRequest
    ) -> MessagesResponse:
        """Create a message response for clients that requested stream=false."""
        stream_request = request_data.model_copy(deep=True)
        stream_request.stream = True
        response = self.create_message(stream_request)
        if isinstance(response, MessagesResponse):
            return response
        if not isinstance(response, StreamingResponse):
            raise HTTPException(status_code=500, detail="Invalid message response")

        chunks: list[str] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode("utf-8", errors="replace"))
            else:
                chunks.append(str(chunk))
        return _messages_response_from_sse_text("".join(chunks))

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as count_tokens_error:
                _log_unexpected_service_exception(
                    self._settings,
                    count_tokens_error,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(
                        count_tokens_error
                    ),
                    detail=get_user_facing_error_message(count_tokens_error),
                ) from count_tokens_error
