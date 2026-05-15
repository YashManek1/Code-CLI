"""OpenAI-style chat base for :class:`OpenAIChatTransport` (NIM, MiniMax, etc.).

``AnthropicMessagesTransport``-based providers (OpenRouter, LM Studio, DeepSeek, …)
live in separate modules; do not list them as subclasses of this class.

MiniMax / weak-model enhancements
----------------------------------
When the requested model's :class:`~tools.ModelQuirks` has ``buffer_tool_calls=True``
(e.g. ``minimax/minimax-m2.7``):

1. Tool argument chunks are **accumulated** across the full stream instead of
   being emitted inline — this prevents truncated / split JSON from reaching
   Claude Code.
2. After the stream closes ``SSEBuilder.emit_buffered_tool_args`` runs
   :func:`~tools.repair_tool_arguments` on every accumulated buffer and emits a
   single valid ``input_json_delta`` per tool.
3. Tool schemas are flattened to ``max_schema_depth`` via
   :func:`~tools.prepare_tools_for_model`` before the upstream request is sent.
4. A ``max_tool_tokens`` cap truncates oversized argument payloads (large file
   writes) and replaces them with an informative error text block so the user
   knows what happened rather than getting a silent crash.

FIX LOG (MiniMax m2.7):
  - _stream_response_impl: buffered path post-stream processing order was wrong.
    Previously: process pending_tool_calls → clear → emit_buffered_tool_args.
    emit_buffered_tool_args then had nothing to iterate because start_tool_block
    was never called (tool states weren't registered yet).
    Fixed: process pending_tool_calls (which calls start_tool_block and
    emit_tool_delta accumulating into raw_arg_buffer) → THEN emit_buffered_tool_args
    → THEN close blocks. pending_tool_calls.clear() moved to AFTER emit.

  - _stream_response_impl: MiniMax sometimes returns a completely empty delta
    with finish_reason="stop" and no content at all (the "no output" bug).
    Added explicit detection: if stream ends with no content blocks started
    and no tool calls, we check accumulated reasoning. If reasoning exists but
    no text, we emit the reasoning as a thinking block + minimal text marker.
    If nothing exists at all, we emit a single-space text to prevent crash.

  - _stream_response_impl: chunk timeout raised from 20s to 60s for MiniMax
    because NIM's M2.7 endpoint has long TTFT (~30-45s) before first token.
    For non-MiniMax models the 20s timeout is preserved.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from core.anthropic import (
    ContentType,
    SSEBuilder,
    ThinkTagParser,
    append_request_id,
    map_stop_reason,
)
from core.anthropic.tools import (
    HeuristicToolParser,
    ModelQuirks,
    get_model_quirks,
    prepare_tools_for_model,
    repair_tool_arguments,
)
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.model_listing import extract_openai_model_ids
from providers.rate_limit import GlobalRateLimiter
import binascii
from datetime import datetime, UTC

from core.healing.stream_manager import HealingStreamManager
from core.healing.recovery_orchestrator import RecoveryOrchestrator
from core.healing.integrity import StreamIntegrityValidator, StreamIntegrityError
from core.healing.normalization import ProviderNormalizationLayer
from core.healing.taxonomy import FailureTaxonomy, FailureType, FailureSeverity
from core.healing.retry_controller import AdaptiveRetryController
from core.healing.engine import HealingContinuationEngine
from core.healing.snapshots import ExecutionSnapshot
from core.healing.poison_detector import ContextPoisonDetector
from core.healing.lifecycle import StabilityAnalytics

# Per-model chunk timeout: MiniMax on NIM has very long TTFT
_DEFAULT_CHUNK_TIMEOUT = 20.0
_SLOW_MODEL_CHUNK_TIMEOUT = 120.0  # NIM MiniMax can take 30-45s before first token
_SLOW_MODEL_FAMILIES = frozenset({"minimax", "kimi", "glm", "moonshot", "qwen3"})


class OpenAIChatTransport(BaseProvider):
    """Base for OpenAI-compatible ``/chat/completions`` adapters (NIM, MiniMax, …)."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        # Initialize persistent healing subsystems
        self._retry_controller = AdaptiveRetryController()
        self._continuation_engine = HealingContinuationEngine()
        self._poison_detector = ContextPoisonDetector()
        self._analytics = StabilityAnalytics()
        self._integrity_validator = StreamIntegrityValidator()
        timeout = httpx.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            read=config.http_read_timeout,
            write=config.http_write_timeout,
            pool=300.0,
        )
        http_client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "limits": httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=300.0,
            ),
            "headers": {
                "Accept-Encoding": "gzip",
                "Connection": "keep-alive",
            },
        }
        if config.proxy:
            http_client_kwargs["proxy"] = config.proxy
        use_http2 = False
        try:
            import h2  # noqa: F401
            use_http2 = True
        except ImportError:
            pass

        http_client = httpx.AsyncClient(
            http2=use_http2,
            **http_client_kwargs,
        )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from the provider's OpenAI-compatible models endpoint."""
        payload = await self._client.models.list()
        return extract_openai_model_ids(payload, provider_name=self._provider_name)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body. Must be implemented by subclasses."""

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning (e.g. OpenRouter reasoning_details)."""
        return iter(())

    def _log_stream_transport_error(
        self, tag: str, req_tag: str, error: Exception
    ) -> None:
        """Log a transport-level stream error safely."""
        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_STREAM_ERROR:{} {}: {}",
                tag,
                req_tag,
                type(error).__name__,
                error,
                exc_info=True,
            )
        else:
            status = getattr(getattr(error, "response", None), "status_code", None)
            logger.error(
                "{}_STREAM_ERROR:{} exc_type={} http_status={}",
                tag,
                req_tag,
                type(error).__name__,
                status,
            )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        return None

    # ------------------------------------------------------------------
    # Stream creation
    # ------------------------------------------------------------------

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming chat completion, optionally retrying once."""
        try:
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **body, stream=True
            )
            return stream, body
        except Exception as error:
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is None:
                raise
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **retry_body, stream=True
            )
            return stream, retry_body

    # ------------------------------------------------------------------
    # Tool argument helpers
    # ------------------------------------------------------------------

    def _emit_tool_arg_delta(
        self, sse: SSEBuilder, tool_call_index: int, args: str, manager: HealingStreamManager | None = None
    ) -> Iterator[str]:
        """Emit one argument fragment for a started tool block.

        - Task tools: buffer until complete JSON, then emit once.
        - Buffer-mode (MiniMax): accumulate silently; emit_buffered_tool_args handles later.
        - Normal: stream raw chunk.
        """
        if not args:
            return
        state = sse.blocks.tool_states.get(tool_call_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = sse.blocks.buffer_task_args(tool_call_index, args)
            if parsed is not None:
                event_to_yield = sse.emit_tool_delta(tool_call_index, json.dumps(parsed))
                if event_to_yield:
                    yield event_to_yield

                if manager:
                    manager._update_checkpoint_from_sse(sse)
            return
        event = sse.emit_tool_delta(tool_call_index, args)
        if event:  # empty string when buffer_tool_calls mode is active
            yield event

    def _process_tool_call(self, tool_call: dict, sse: SSEBuilder, manager: HealingStreamManager | None = None) -> Iterator[str]:
        """Process a single tool call delta and yield SSE events."""
        tool_call_index = tool_call.get("index", 0)
        if tool_call_index < 0:
            tool_call_index = len(sse.blocks.tool_states)

        function_delta = tool_call.get("function", {})
        incoming_name = function_delta.get("name")
        raw_arguments = function_delta.get("arguments")

        if raw_arguments is None:
            arguments = ""
        elif isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments, ensure_ascii=False)

        logger.debug(
            "TOOL_CALL_DELTA index={} id={} name={} args_len={}",
            tool_call.get("index", 0),
            tool_call.get("id"),
            incoming_name,
            len(arguments),
        )

        if tool_call.get("id") is not None:
            sse.blocks.set_stream_tool_id(tool_call_index, tool_call.get("id"))

        if incoming_name is not None:
            sse.blocks.register_tool_name(tool_call_index, incoming_name)

        state = sse.blocks.tool_states.get(tool_call_index)
        resolved_id = (
            state.tool_id if state and state.tool_id else None
        ) or tool_call.get("id")
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                yield sse.start_tool_block(tool_call_index, tool_id, display_name)
                state = sse.blocks.tool_states[tool_call_index]
                if state.pre_start_args:
                    pre_start_arguments = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(
                        sse, tool_call_index, pre_start_arguments, manager=manager
                    )

        state = sse.blocks.tool_states.get(tool_call_index)
        if not arguments:
            return
        if state is None or not state.started:
            state = sse.blocks.ensure_tool_state(tool_call_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(sse, tool_call_index, arguments, manager=manager)

    def _flush_task_arg_buffers(self, sse: SSEBuilder) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta (best-effort)."""
        for tool_index, out in sse.blocks.flush_task_arg_buffers():
            event = sse.emit_tool_delta(tool_index, out)
            if event:
                yield event

    # ------------------------------------------------------------------
    # Pending tool-call accumulation helpers (MiniMax buffered path)
    # ------------------------------------------------------------------

    def _validate_and_repair_pending_tool(
        self,
        tc_info: dict,
        model: str,
    ) -> dict | None:
        """Validate / repair one fully-accumulated tool call.

        Returns the (possibly mutated) ``tc_info`` dict on success, or
        ``None`` if the call should be discarded entirely.
        """
        function_data = tc_info.get("function", {})
        arguments = function_data.get("arguments", "")

        if not isinstance(arguments, str):
            logger.warning(
                "Skipping tool call with non-string arguments (type={})",
                type(arguments).__name__,
            )
            return None

        arguments = arguments.strip()

        if not arguments:
            logger.warning(
                "Skipping tool call '{}' with empty arguments",
                function_data.get("name", "?"),
            )
            return None

        # Quick parse attempt first (avoids repair overhead for healthy models)
        parsed = self._safe_json_loads(arguments)

        if parsed is not None:
            function_data["arguments"] = json.dumps(
                parsed,
                ensure_ascii=False,
            )
            return tc_info

        # Invoke multi-strategy repair
        tool_name = function_data.get("name", "")
        repaired = repair_tool_arguments(arguments, tool_name=tool_name, model=model)
        if repaired == "{}":
            logger.warning(
                "JSON_REPAIR: tool '{}' args could not be repaired, emitting {{}}",
                tool_name,
            )
        function_data["arguments"] = repaired
        return tc_info

    @staticmethod
    def _check_tool_token_cap(tc_info: dict, quirks: ModelQuirks) -> bool:
        """Return False only for dangerous oversized non-file-edit tools."""

        cap = quirks.max_tool_tokens

        if not cap:
            return True

        function_data = tc_info.get("function", {})
        tool_name = function_data.get("name", "?")
        arguments = function_data.get("arguments", "")

        argument_length = len(arguments)

        # File editing tools legitimately need huge payloads.
        # Allow them through unless they become absurdly large.
        if tool_name in {"Write", "Edit", "MultiEdit"}:
            hard_cap = cap * 4

            if argument_length > hard_cap:
                logger.warning(
                    "TOKEN_CAP_HARD_LIMIT: tool '{}' args {} chars > hard_cap {}",
                    tool_name,
                    argument_length,
                    hard_cap,
                )
                return False

            logger.info(
                "TOKEN_CAP_BYPASS: allowing large '{}' payload ({} chars)",
                tool_name,
                argument_length,
            )
            return True

        if argument_length > cap:
            logger.warning(
                "TOKEN_CAP: tool '{}' args {} chars > cap {}; skipping",
                tool_name,
                argument_length,
                cap,
            )
            return False

        return True

    def _is_tool_call_complete(
        self,
        tc_info: dict,
    ) -> bool:
        function_data = tc_info.get("function", {})
        tool_name = function_data.get("name", "")
        arguments = function_data.get("arguments", "{}")

        try:
            parsed = self._safe_json_loads(arguments)
        except Exception:
            return False

        if not isinstance(parsed, dict):
            return False

        if tool_name == "Write":
            return "file_path" in parsed and "content" in parsed

        if tool_name == "Edit":
            return (
                "file_path" in parsed
                and "old_string" in parsed
                and "new_string" in parsed
            )

        if tool_name == "MultiEdit":
            return "file_path" in parsed and "edits" in parsed

        if tool_name == "Task":
            return "description" in parsed

        return True

    def _safe_json_loads(self, raw: str) -> dict | list | None:
        """Safely parse provider JSON fragments."""
        if not raw:
            return None

        if not isinstance(raw, str):
            return None

        raw = raw.strip()

        if not raw:
            return None

        if raw == "[DONE]":
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(
                "STREAM_JSON_SKIP: malformed/empty JSON fragment len={}",
                len(raw),
            )
            return None

    def _normalize_delta_content(
        self,
        content: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Normalize provider delta.content into:
        1. plain text string
        2. synthetic tool_calls list

        Handles providers that emit:
        - plain strings
        - OpenAI structured content arrays
        - Anthropic-style content blocks
        - mixed text/tool arrays
        - Python-repr artifact leaks (Kimi/MiniMax)

        Prevents stream stalls from list-based content payloads.
        """
        if content is None:
            return "", []

        if isinstance(content, str):
            # Strip leaked tool-call protocol tokens
            tool_protocol_tokens = [
                "<|tool_call_begin|>",
                "<|tool_call_end|>",
                "<|tool_calls_section_begin|>",
                "<|tool_calls_section_end|>",
                "<|tool_call_argument_begin|>",
                "<|tool_call_argument_end|>",
            ]
            for token in tool_protocol_tokens:
                content = content.replace(token, "")

            stripped_content = content.strip()

            # Handle Kimi/MiniMax "artifact leaks" where they emit parts of a list wrapper
            # or Python-repr strings as plain text.
            if stripped_content in ("[", "]", "[]"):
                return "", []

            # Python-repr list wrapper: "[{'type':'text',..."
            if stripped_content.startswith("[{'type':"):
                try:
                    parsed = ast.literal_eval(stripped_content)
                    if isinstance(parsed, list):
                        return self._normalize_delta_content(parsed)
                except Exception:
                    # Fallback for malformed reprs (e.g. unclosed inner lists)
                    matches = list(
                        re.finditer(
                            r"['\"]text['\"]\s*:\s*(?P<quote>['\"])",
                            stripped_content,
                        )
                    )
                    if matches:
                        last_match = matches[-1]
                        quote_character = last_match.group("quote")
                        text_start_index = last_match.end()
                        text_end_index = -1
                        cursor_index = text_start_index
                        while cursor_index < len(stripped_content):
                            previous_character = stripped_content[
                                cursor_index - 1 : cursor_index
                            ]
                            current_character = stripped_content[cursor_index]
                            if (
                                current_character == quote_character
                                and previous_character != "\\"
                            ):
                                text_end_index = cursor_index
                                break
                            cursor_index += 1
                        if text_end_index != -1:
                            unquoted_text = stripped_content[
                                text_start_index:text_end_index
                            ]
                        else:
                            unquoted_text = stripped_content[text_start_index:].rstrip(
                                "'] }\""
                            )
                        return self._normalize_delta_content(
                            unquoted_text.replace("\\'", "'").replace('\\"', '"')
                        )

            # Dangling leading bracket or assistant prefix
            if stripped_content.startswith("[") and not stripped_content.endswith("]"):
                if stripped_content.startswith("[assistant]"):
                    return stripped_content.replace("[assistant]", "").strip(), []
                return stripped_content[1:], []

            # Some OpenAI-compatible providers incorrectly serialize
            # structured content arrays into JSON strings.
            if stripped_content.startswith("[") and stripped_content.endswith("]"):
                try:
                    parsed_content = json.loads(stripped_content)
                    if isinstance(parsed_content, list):
                        return self._normalize_delta_content(parsed_content)
                except Exception:
                    pass

            return content, []

        if not isinstance(content, list):
            return str(content), []

        text_parts: list[str] = []
        synthetic_tool_calls: list[dict[str, Any]] = []

        for item in content:
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue

            item_type = item.get("type")

            # Standard text blocks
            if item_type in {"text", "output_text"}:
                text_value = item.get("text", "")
                if text_value:
                    # Recursive normalize in case the text is itself a leaked wrapper
                    norm_text, _ = self._normalize_delta_content(text_value)
                    text_parts.append(norm_text)
                continue

            # Some providers emit reasoning separately
            if item_type in {"thinking", "reasoning"}:
                reasoning_text = (
                    item.get("thinking")
                    or item.get("reasoning")
                    or item.get("text")
                    or ""
                )
                if reasoning_text:
                    text_parts.append(str(reasoning_text))
                continue

            # Tool calls embedded directly inside content array
            if item_type in {"tool_use", "tool_call"}:
                tool_name = item.get("name") or item.get("tool_name") or "tool_call"
                tool_input = item.get("input") or item.get("arguments") or {}

                if not isinstance(tool_input, str):
                    tool_input = json.dumps(
                        tool_input,
                        ensure_ascii=False,
                    )

                synthetic_tool_calls.append(
                    {
                        "index": len(synthetic_tool_calls),
                        "id": item.get("id") or f"tool_{uuid.uuid4()}",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_input,
                        },
                    }
                )
                continue

            # Fallback text extraction
            fallback_text = item.get("text") or item.get("content") or ""
            if fallback_text:
                text_parts.append(str(fallback_text))

        return "".join(text_parts), synthetic_tool_calls

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format with full Healing Loop Architecture."""
        session_id = getattr(request, "session_id", "default")
        provider_name = self._provider_name
        
        # 1. Access Healing Subsystems
        manager = HealingStreamManager(session_id, request_id)
        integrity = self._integrity_validator
        retry_controller = self._retry_controller
        continuation_engine = self._continuation_engine
        poison_detector = self._poison_detector
        analytics = self._analytics
        
        state_store = getattr(self._config, "execution_state_store", None)
        state = state_store.load(session_id) if state_store else None
        
        with logger.contextualize(request_id=request_id):
            max_healing_attempts = 3
            current_request = request
            
            for attempt in range(max_healing_attempts):
                try:
                    # 2. Wrapped Execution Loop
                    async for event in manager.wrap_stream(
                        self._stream_response_impl(
                            current_request, input_tokens, request_id, 
                            thinking_enabled=thinking_enabled, manager=manager
                        )
                    ):
                        # Verify integrity before yielding
                        # (In a real impl, we'd parse the SSE event into a dict first)
                        # integrity.verify_chunk(event, manager.event_count)
                        yield event
                    
                    # Success: Record analytics and exit
                    analytics.record_event("successful_healing" if attempt > 0 else "total_sessions")
                    return

                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception as exc:
                    analytics.record_event("stream_interruptions")
                    
                    # 3. Detect & Classify
                    failure_type = FailureTaxonomy.classify(exc)
                    logger.warning("HEALING: Detected {} (attempt {}/{})", failure_type.type, attempt + 1, max_healing_attempts)
                    
                    if attempt >= max_healing_attempts - 1:
                        logger.error("HEALING: All attempts exhausted for session={}", session_id)
                        raise

                    # 4. Check for Cognition Poisoning
                    if state and poison_detector.is_poisoned(state.retry_history, state.validation_failures):
                        logger.error("HEALING: Loop collapse detected. Aborting to prevent context poisoning.")
                        raise

                    # 5. Snapshot & Log Failure
                    if state:
                        snapshot = ExecutionSnapshot(
                            snapshot_id=f"snap_{uuid.uuid4().hex[:8]}",
                            parent_snapshot_id=getattr(state, "current_snapshot_id", None),
                            state=state,
                            reason=f"failure_{failure_type.type.lower()}"
                        )
                        state.retry_history.append({
                            "timestamp": datetime.now(UTC).isoformat(),
                            "error": str(exc),
                            "failure_type": failure_type.type,
                            "snapshot_id": snapshot.snapshot_id
                        })
                        if state_store:
                            state_store.save(state)

                    # 6. Reconstruct & Continue (Healing Engine)
                    resumption_state = manager.get_resumption_state()
                    new_messages = continuation_engine.build_resumption_messages(
                        getattr(current_request, "messages", []),
                        resumption_state.get("content", ""),
                        resumption_state.get("tool_calls", []),
                        failure_context=failure_type.type
                    )
                    
                    # 7. Adaptive Backoff
                    await retry_controller.wait_before_retry(provider_name, failure_type)

                    # 8. Update request for next attempt
                    if isinstance(current_request, list):
                        current_request = new_messages
                    else:
                        current_request.messages = new_messages

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
        manager: HealingStreamManager | None = None,
    ) -> AsyncIterator[str]:
        """Shared streaming implementation with MiniMax / weak-model hardening."""
        tag = self._provider_name
        model_str: str = getattr(request, "model", "") or ""
        quirks: ModelQuirks = get_model_quirks(model_str)

        # FIX: MiniMax on NIM has very long TTFT (30-45s). Use a longer timeout
        # for MiniMax models to prevent premature keepalive loops.
        is_slow_model = any(f in model_str.lower() for f in _SLOW_MODEL_FAMILIES)
        chunk_timeout = (
            _SLOW_MODEL_CHUNK_TIMEOUT if is_slow_model else _DEFAULT_CHUNK_TIMEOUT
        )

        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(
            message_id,
            model_str,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
            buffer_tool_calls=quirks.buffer_tool_calls,
        )

        body = self._build_request_body(request, thinking_enabled=thinking_enabled)

        # Flatten tool schemas for weak models before sending upstream
        if quirks.flatten_tool_schemas and "tools" in body:
            body["tools"] = prepare_tools_for_model(body["tools"], model_str)
            logger.info(
                "SCHEMA_FLATTEN: Applied schema flattening for model '{}'", model_str
            )

        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "{}_STREAM:{} model={} msgs={} tools={} buffer_tool_calls={} quirks_repair={} chunk_timeout={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
            quirks.buffer_tool_calls,
            quirks.requires_json_repair,
            chunk_timeout,
        )

        yield sse.message_start()
        
        if manager:
            manager._update_checkpoint_from_sse(sse)

        heuristic_parser = HeuristicToolParser(model=model_str)
        think_parser = ThinkTagParser()
        finish_reason = None
        usage_info = None
        # Accumulate all tool call chunks keyed by OpenAI index.
        # Emitted all at once after finish_reason arrives (standard) or at
        # stream end (MiniMax buffered mode where we skip inline delta).
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        _FAMILY_LIMITER_PATTERNS = (
            ("minimax", "minimax"),
            ("kimi", "kimi"),
            ("glm", "glm"),
            ("moonshot", "moonshot"),
            ("qwen", "qwen"),
            ("deepseek", "deepseek"),
            ("mistral", "mistral"),
        )
        model_family = next(
            (
                family
                for pattern, family in _FAMILY_LIMITER_PATTERNS
                if pattern in model_str.lower()
            ),
            None,
        )
        if model_family:
            effective_limiter = GlobalRateLimiter.get_scoped_instance(
                f"{self._provider_name.lower()}_{model_family}",
                rate_limit=self._config.rate_limit,
                rate_window=self._config.rate_window,
                max_concurrency=self._config.max_concurrency,
            )
        else:
            effective_limiter = self._global_rate_limiter

        async with effective_limiter.concurrency_slot():
            try:
                stream, body = await self._create_stream(body)
                last_provider_chunk_at = time.monotonic()

                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(),
                            timeout=chunk_timeout,
                        )
                        last_provider_chunk_at = time.monotonic()
                        if chunk is None:
                            logger.warning("STREAM_SKIP: received None chunk")
                            continue
                    except TimeoutError:
                        idle_s = time.monotonic() - last_provider_chunk_at
                        effective_idle_timeout = max(
                            self._config.stream_idle_timeout,
                            chunk_timeout * 1.5,
                        )
                        if idle_s >= effective_idle_timeout:
                            logger.warning(
                                "{}_STREAM: upstream idle timeout {:.1f}s > {:.1f}s{}",
                                tag,
                                idle_s,
                                effective_idle_timeout,
                                req_tag,
                            )
                            error_message = append_request_id(
                                (
                                    f"{tag} stream stalled for "
                                    f"{effective_idle_timeout:.0f}s without "
                                    "provider chunks."
                                ),
                                request_id,
                            )
                            for event in sse.close_all_blocks():
                                yield event
                            if sse.blocks.has_emitted_tool_block():
                                yield sse.emit_top_level_error(error_message)
                            else:
                                for event in sse.emit_error(error_message):
                                    yield event
                            yield sse.message_delta("end_turn", 1)
                            yield sse.message_stop()
                            return
                        logger.warning(
                            "{}_STREAM: waiting for upstream chunk (timeout={}s idle={:.1f}s)",
                            tag,
                            chunk_timeout,
                            idle_s,
                        )
                        continue

                    except StopAsyncIteration:
                        break
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage

                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        logger.debug("STREAM_SKIP: empty choices")
                        continue

                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)

                    if delta is None:
                        logger.debug("STREAM_SKIP: null delta")
                        continue

                    logger.debug("RAW_DELTA model={} delta={}", model_str, delta)

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.debug("{} finish_reason: {}", tag, finish_reason)

                    # ---------------------------------------------------
                    # Reasoning content
                    # ---------------------------------------------------
                    reasoning = getattr(delta, "reasoning_content", None)
                    if thinking_enabled and reasoning:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(reasoning)

                    for event in self._handle_extra_reasoning(
                        delta, sse, thinking_enabled=thinking_enabled
                    ):
                        yield event

                    # ---------------------------------------------------
                    # Text content
                    # ---------------------------------------------------
                    raw_content = getattr(delta, "content", None)

                    normalized_text, synthetic_tool_calls = (
                        self._normalize_delta_content(raw_content)
                    )

                    if normalized_text:
                        # Heuristic tool detection for Kimi/MiniMax inline text emission
                        safe_text, heuristic_tool_calls = heuristic_parser.feed(
                            normalized_text
                        )
                        if heuristic_tool_calls:
                            # Convert Anthropic-style to OpenAI-style for the transport's accumulation loop
                            for htc in heuristic_tool_calls:
                                synthetic_tool_calls.append(
                                    {
                                        "index": len(synthetic_tool_calls),
                                        "id": htc.get("id"),
                                        "function": {
                                            "name": htc.get("name"),
                                            "arguments": json.dumps(
                                                htc.get("input", {}), ensure_ascii=False
                                            ),
                                        },
                                    }
                                )

                        for part in think_parser.feed(safe_text):
                            raw_part_content = part.content
                            if isinstance(raw_part_content, list):
                                normalized_part_content = json.dumps(
                                    raw_part_content,
                                    ensure_ascii=False,
                                )
                            else:
                                normalized_part_content = str(raw_part_content)

                            if not normalized_part_content.strip():
                                continue

                            if part.type == ContentType.THINKING:
                                if not thinking_enabled:
                                    continue
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(normalized_part_content)
                            else:
                                # FIX: Don't skip text even if tools are pending.
                                # SSEBuilder handles proper interleaving.
                                for event in sse.ensure_text_block():
                                    yield event
                                yield sse.emit_text_delta(normalized_part_content)

                    # Merge synthetic tool calls emitted via structured content arrays
                    if synthetic_tool_calls:
                        existing_tool_calls = getattr(delta, "tool_calls", None)
                        if existing_tool_calls:
                            tool_calls = (
                                list(existing_tool_calls) + synthetic_tool_calls
                            )
                        else:
                            tool_calls = synthetic_tool_calls
                    else:
                        tool_calls = getattr(delta, "tool_calls", None)

                    # ---------------------------------------------------
                    # Native tool calls - accumulate chunks
                    # ---------------------------------------------------
                    if tool_calls:
                        for tool_call_delta in tool_calls:
                            # Normalize: handle both OpenAI objects and synthetic dicts
                            is_dict = isinstance(tool_call_delta, dict)
                            tool_call_index = (
                                tool_call_delta.get("index", 0)
                                if is_dict
                                else getattr(tool_call_delta, "index", 0)
                            )

                            existing = accumulated_tool_calls.setdefault(
                                tool_call_index,
                                {
                                    "index": tool_call_index,
                                    "id": None,
                                    "function": {"name": "", "arguments": ""},
                                },
                            )

                            tool_call_id = (
                                tool_call_delta.get("id")
                                if is_dict
                                else getattr(tool_call_delta, "id", None)
                            )
                            if tool_call_id:
                                existing["id"] = tool_call_id

                            tool_call_function = (
                                tool_call_delta.get("function")
                                if is_dict
                                else getattr(tool_call_delta, "function", None)
                            )
                            if tool_call_function:
                                function_name = (
                                    tool_call_function.get("name")
                                    if is_dict
                                    else getattr(tool_call_function, "name", None)
                                )
                                function_arguments = (
                                    tool_call_function.get("arguments")
                                    if is_dict
                                    else getattr(tool_call_function, "arguments", None)
                                )
                                if function_name:
                                    existing["function"]["name"] = function_name
                                if function_arguments:
                                    current_args = existing["function"]["arguments"]
                                    if not current_args:
                                        existing["function"]["arguments"] = (
                                            function_arguments
                                        )
                                    else:
                                        # If chunk looks like a complete JSON block, it might be a replacement.
                                        # But usually we just append to preserve fragments.
                                        if function_arguments.strip().startswith(
                                            "{"
                                        ) and function_arguments.strip().endswith("}"):
                                            existing["function"]["arguments"] = (
                                                function_arguments
                                            )
                                        else:
                                            existing["function"]["arguments"] = (
                                                current_args + function_arguments
                                            )

                    # ---------------------------------------------------
                    # FIX: Flush pending tool calls AFTER accumulation.
                    # This ensures the last chunk of arguments is captured.
                    # ---------------------------------------------------
                    if accumulated_tool_calls and not quirks.buffer_tool_calls:
                        completed_indices = []
                        for (
                            tool_call_index,
                            tool_call_info,
                        ) in accumulated_tool_calls.items():
                            arguments = (
                                tool_call_info.get("function", {})
                                .get("arguments", "")
                                .strip()
                            )
                            if not arguments:
                                continue
                            parsed = self._safe_json_loads(arguments)
                            if parsed is None:
                                continue
                            completed_indices.append(tool_call_index)

                        if completed_indices:
                            for event in sse.close_content_blocks():
                                yield event

                        for tool_call_index in completed_indices:
                            tool_call_info = accumulated_tool_calls.pop(tool_call_index)
                            repaired = self._validate_and_repair_pending_tool(
                                tool_call_info,
                                model_str,
                            )
                            if repaired is None:
                                continue
                            if not self._is_tool_call_complete(repaired):
                                logger.warning(
                                    "Skipping incomplete tool call '{}'",
                                    repaired.get("function", {}).get("name", "?"),
                                )
                                continue

                            if not self._check_tool_token_cap(repaired, quirks):
                                continue

                            for event in self._process_tool_call(repaired, sse, manager=manager):
                                yield event

            except (asyncio.CancelledError, GeneratorExit):
                raise
            except (
                json.JSONDecodeError,
                ValueError,
            ) as parse_error:
                logger.warning(
                    "STREAM_RECOVERABLE_PARSE_ERROR: {}",
                    repr(parse_error),
                )

                for event in sse.close_all_blocks():
                    yield event

                for event in sse.emit_error(
                    append_request_id(
                        "Recoverable provider stream parse error.",
                        request_id,
                    )
                ):
                    yield event

                yield sse.message_delta("end_turn", 1)
                yield sse.message_stop()
                return

        # ==================================================================
        # Post-stream processing
        # ==================================================================

        # ------------------------------------------------------------------
        # Buffered path (MiniMax): process all accumulated tool calls at once.
        #
        # FIX: Original order was:
        #   1. process accumulated_tool_calls (calls _process_tool_call → start_tool_block
        #      → emit_tool_delta which accumulates into raw_arg_buffer)
        #   2. accumulated_tool_calls.clear()   ← this happened BEFORE emit_buffered_tool_args
        #   3. emit_buffered_tool_args      ← but tool states existed so it worked
        #
        # The real bug was that start_tool_block was being called for each pending
        # tool, then emit_buffered_tool_args tried to emit again — causing duplicate
        # content_block_start events. The buffered path should NOT call
        # _process_tool_call for argument streaming at all; it should only
        # call start_tool_block and let emit_buffered_tool_args handle the args.
        #
        # New correct order:
        #   1. For each pending tool: validate/repair args, check cap, start tool block
        #      (do NOT call _process_tool_call which would also emit args inline)
        #   2. emit_buffered_tool_args (emits single repaired arg delta per tool)
        #   3. clear accumulated_tool_calls
        # ------------------------------------------------------------------
        if quirks.buffer_tool_calls and accumulated_tool_calls:
            for event in sse.close_content_blocks():
                yield event

            for tool_call_info in list(accumulated_tool_calls.values()):
                repaired = self._validate_and_repair_pending_tool(
                    tool_call_info, model_str
                )
                if repaired is None:
                    continue
                if not self._is_tool_call_complete(repaired):
                    logger.warning(
                        "Skipping incomplete buffered tool call '{}'",
                        repaired.get("function", {}).get("name", "?"),
                    )
                    continue
                if not self._check_tool_token_cap(repaired, quirks):
                    # Replace oversized write with an informative text error
                    tool_name = tool_call_info.get("function", {}).get("name", "tool")
                    for event in sse.ensure_text_block():
                        yield event
                    yield sse.emit_text_delta(
                        f"\n\n[Proxy: {tool_name} arguments exceeded the "
                        f"{quirks.max_tool_tokens:,}-character cap and were skipped. "
                        "Consider breaking the operation into smaller chunks.]\n"
                    )
                    continue

                # FIX: In buffered mode, only start the tool block here.
                # Do NOT call _process_tool_call() because that would emit args
                # inline AND accumulate them — causing double emission.
                # Instead, store the repaired args in the tool state's raw_arg_buffer
                # so emit_buffered_tool_args picks them up as a single clean delta.
                tool_call_index = repaired.get("index", 0)
                tool_call_id = repaired.get("id") or f"tool_{uuid.uuid4()}"
                tool_name = repaired.get("function", {}).get("name", "") or "tool_call"
                repaired_args = repaired.get("function", {}).get("arguments", "{}")

                # Register tool state if not already done during stream
                if (
                    tool_call_index not in sse.blocks.tool_states
                    or not sse.blocks.tool_states[tool_call_index].started
                ):
                    yield sse.start_tool_block(
                        tool_call_index, str(tool_call_id), tool_name
                    )

                # Store repaired args in raw_arg_buffer for emit_buffered_tool_args
                state = sse.blocks.tool_states.get(tool_call_index)
                if state is not None and not state.buffered_args_emitted:
                    # Overwrite raw_arg_buffer with the fully repaired JSON
                    # (replaces any partial chunks accumulated during stream)
                    state.raw_arg_buffer = repaired_args

            # FIX: emit_buffered_tool_args BEFORE clearing accumulated_tool_calls
            for event in sse.emit_buffered_tool_args(model_str):
                yield event

            accumulated_tool_calls.clear()

        # ------------------------------------------------------------------
        # Flush remaining heuristic-parser and think-parser content
        # ------------------------------------------------------------------
        heuristic_extra = heuristic_parser.flush()
        if heuristic_extra:
            synthetic_tool_calls.extend(heuristic_extra)

        remaining = think_parser.flush()
        if remaining and remaining.type == ContentType.THINKING and thinking_enabled:
            for event in sse.ensure_thinking_block():
                yield event
            yield sse.emit_thinking_delta(remaining.content)
        elif remaining and remaining.type == ContentType.TEXT:
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(remaining.content)
            # Final emergency flush for pending tool calls
        if accumulated_tool_calls:
            logger.warning(
                "FINAL_TOOL_FLUSH: emitting {} pending tool calls",
                len(accumulated_tool_calls),
            )
            for event in sse.close_content_blocks():
                yield event
            for tool_call_info in list(accumulated_tool_calls.values()):
                repaired = self._validate_and_repair_pending_tool(
                    tool_call_info,
                    model_str,
                )
                if repaired is None:
                    continue
                if not self._is_tool_call_complete(repaired):
                    logger.warning(
                        "Skipping incomplete emergency tool call '{}'",
                        repaired.get("function", {}).get("name", "?"),
                    )
                    continue
                if not self._check_tool_token_cap(repaired, quirks):
                    continue
                for event in self._process_tool_call(repaired, sse, manager=manager):
                    yield event
            accumulated_tool_calls.clear()
        # ------------------------------------------------------------------
        # Ensure at least one content block so Claude Code doesn't crash.
        #
        # FIX: MiniMax M2.7 on NIM sometimes returns a stream with ONLY
        # reasoning content and no text/tool content at all (the "no output" bug).
        # When this happens accumulated_reasoning is non-empty but there are no
        # content blocks. We must emit at least a text block to prevent a crash.
        # ------------------------------------------------------------------
        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
        has_content_blocks = (
            sse.blocks.text_index != -1
            or sse.blocks.thinking_index != -1
            or has_started_tool
        )

        if not has_content_blocks:
            # FIX: If we have accumulated reasoning but no visible output,
            # emit reasoning as a thinking block (if thinking is enabled) plus
            # a minimal text marker so Claude Code knows something happened.
            accumulated_reasoning = sse.accumulated_reasoning
            if accumulated_reasoning and thinking_enabled:
                for event in sse.ensure_thinking_block():
                    yield event
                yield sse.emit_thinking_delta(accumulated_reasoning)
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")
        elif (
            not has_started_tool
            and not sse.accumulated_text.strip()
            and sse.accumulated_reasoning.strip()
        ):
            # Reasoning-only response: emit minimal text block
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        # ------------------------------------------------------------------
        # Task-arg buffer flush (non-buffered path only)
        # ------------------------------------------------------------------
        if not quirks.buffer_tool_calls:
            for event in self._flush_task_arg_buffers(sse):
                yield event

        # ------------------------------------------------------------------
        # Close all open blocks
        # ------------------------------------------------------------------
        for event in sse.close_all_blocks():
            yield event

        # ------------------------------------------------------------------
        # Final message_delta / message_stop
        # ------------------------------------------------------------------
        completion = (
            getattr(usage_info, "completion_tokens", None)
            if usage_info is not None
            else None
        )
        output_tokens = (
            completion if isinstance(completion, int) else sse.estimate_output_tokens()
        )
        if usage_info and hasattr(usage_info, "prompt_tokens"):
            provider_input = usage_info.prompt_tokens
            if isinstance(provider_input, int):
                logger.debug(
                    "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                    input_tokens,
                    provider_input,
                    provider_input - input_tokens,
                )
        # Force correct stop reason if tools were emitted
        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
        if not finish_reason:
            if has_started_tool:
                finish_reason = "tool_calls"
            else:
                logger.warning("STREAM_RECOVERY: missing finish_reason, forcing stop")
                finish_reason = "stop"
        elif finish_reason == "stop" and has_started_tool:
            finish_reason = "tool_calls"

        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
        return
