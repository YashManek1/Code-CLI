"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .execution_state_store import ExecutionStateStore
from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, SystemContent, TokenCountRequest
from .models.responses import TokenCountResponse
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

    def _extract_session_id(self, request_data: MessagesRequest) -> str | None:
        """Extract session_id from forwarded headers or metadata."""
        # Check forwarded headers
        if request_data.forwarded_headers:
            for header_name in ("x-session-id", "anthropic-conversation-id"):
                value = request_data.forwarded_headers.get(header_name)
                if value:
                    return value
        # Check metadata
        if request_data.metadata:
            session_id = request_data.metadata.get("session_id")
            if session_id:
                return str(session_id)
        return None

    def _inject_execution_state(
        self, request_data: MessagesRequest, session_id: str | None
    ) -> None:
        """Inject execution state context into the request's system field."""
        if self._execution_state_store is None or session_id is None:
            return

        state = self._execution_state_store.load(session_id)
        if state is None:
            return

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
            request_data.system = [state_block] + list(request_data.system)

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

            # Inject execution state context before routing
            session_id = self._extract_session_id(request_data)
            self._inject_execution_state(request_data, session_id)
            
            # Process tool results to advance the orchestration engine
            if self._execution_state_store is not None and session_id is not None:
                from .orchestration.response_tracker import ResponseTracker
                tracker = ResponseTracker(self._execution_state_store)
                tracker.process_request_messages(session_id, request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
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
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

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
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
