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

import asyncio
import json
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
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.model_listing import extract_openai_model_ids
from providers.rate_limit import GlobalRateLimiter
from core.anthropic.tools import (
    ModelQuirks,
    get_model_quirks,
    prepare_tools_for_model,
    repair_tool_arguments,
)

# Per-model chunk timeout: MiniMax on NIM has very long TTFT
_DEFAULT_CHUNK_TIMEOUT = 20.0
_MINIMAX_CHUNK_TIMEOUT = 90.0  # NIM MiniMax can take 30-45s before first token


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
        timeout = httpx.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            read=config.http_read_timeout,
            write=config.http_write_timeout,
            pool=300.0,
        )
        http_client = httpx.AsyncClient(
            proxy=config.proxy or None,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=120
            ),
            headers={
                "Accept-Encoding": "gzip",
                "Connection": "keep-alive",
            },
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
        self, sse: SSEBuilder, tc_index: int, args: str
    ) -> Iterator[str]:
        """Emit one argument fragment for a started tool block.

        - Task tools: buffer until complete JSON, then emit once.
        - Buffer-mode (MiniMax): accumulate silently; emit_buffered_tool_args handles later.
        - Normal: stream raw chunk.
        """
        if not args:
            return
        state = sse.blocks.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = sse.blocks.buffer_task_args(tc_index, args)
            if parsed is not None:
                event = sse.emit_tool_delta(tc_index, json.dumps(parsed))
                if event:
                    yield event
            return
        event = sse.emit_tool_delta(tc_index, args)
        if event:  # empty string when buffer_tool_calls mode is active
            yield event

    def _process_tool_call(self, tc: dict, sse: SSEBuilder) -> Iterator[str]:
        """Process a single tool call delta and yield SSE events."""
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        raw_arguments = fn_delta.get("arguments")

        if raw_arguments is None:
            arguments = ""
        elif isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments, ensure_ascii=False)

        logger.debug(
            "TOOL_CALL_DELTA index={} id={} name={} args_len={}",
            tc.get("index", 0),
            tc.get("id"),
            incoming_name,
            len(arguments),
        )

        if tc.get("id") is not None:
            sse.blocks.set_stream_tool_id(tc_index, tc.get("id"))

        if incoming_name is not None:
            sse.blocks.register_tool_name(tc_index, incoming_name)

        state = sse.blocks.tool_states.get(tc_index)
        resolved_id = (state.tool_id if state and state.tool_id else None) or tc.get(
            "id"
        )
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                yield sse.start_tool_block(tc_index, tool_id, display_name)
                state = sse.blocks.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(sse, tc_index, pre)

        state = sse.blocks.tool_states.get(tc_index)
        if not arguments:
            return
        if state is None or not state.started:
            state = sse.blocks.ensure_tool_state(tc_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(sse, tc_index, arguments)

    def _flush_task_arg_buffers(self, sse: SSEBuilder) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta (best-effort)."""
        for tool_index, out in sse.blocks.flush_task_arg_buffers():
            event = sse.emit_tool_delta(tool_index, out)
            if event:
                yield event

    # ------------------------------------------------------------------
    # Pending tool-call accumulation helpers (MiniMax buffered path)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_repair_pending_tool(tc_info: dict, model: str) -> dict | None:
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
        try:
            parsed = json.loads(arguments)
            function_data["arguments"] = json.dumps(parsed, ensure_ascii=False)
            return tc_info
        except json.JSONDecodeError:
            pass

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
        """Return False (and log) if argument JSON exceeds ``max_tool_tokens``."""
        cap = quirks.max_tool_tokens
        if not cap:
            return True
        args = tc_info.get("function", {}).get("arguments", "")
        if len(args) > cap:
            tool_name = tc_info.get("function", {}).get("name", "?")
            logger.warning(
                "TOKEN_CAP: tool '{}' args {} chars > cap {}; skipping",
                tool_name,
                len(args),
                cap,
            )
            return False
        return True
    
    @staticmethod
    def _is_tool_call_complete(tc_info: dict) -> bool:
        function_data = tc_info.get("function", {})
        tool_name = function_data.get("name", "")
        arguments = function_data.get("arguments", "{}")

        try:
            parsed = json.loads(arguments)
        except Exception:
            return False

        if not isinstance(parsed, dict):
            return False

        if tool_name == "Write":
            return (
                "file_path" in parsed
                and "content" in parsed
            )

        if tool_name == "Edit":
            return (
                "file_path" in parsed
                and "old_string" in parsed
                and "new_string" in parsed
            )

        if tool_name == "MultiEdit":
            return (
                "file_path" in parsed
                and "edits" in parsed
            )

        if tool_name == "Task":
            return "description" in parsed

        return True
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
        """Stream response in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id, thinking_enabled=thinking_enabled
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """Shared streaming implementation with MiniMax / weak-model hardening."""
        tag = self._provider_name
        model_str: str = getattr(request, "model", "") or ""
        quirks: ModelQuirks = get_model_quirks(model_str)

        # FIX: MiniMax on NIM has very long TTFT (30-45s). Use a longer timeout
        # for MiniMax models to prevent premature keepalive loops.
        is_minimax = "minimax" in model_str.lower()
        chunk_timeout = _MINIMAX_CHUNK_TIMEOUT if is_minimax else _DEFAULT_CHUNK_TIMEOUT

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

        think_parser = ThinkTagParser()
        finish_reason = None
        usage_info = None
        # Accumulate all tool call chunks keyed by OpenAI index.
        # Emitted all at once after finish_reason arrives (standard) or at
        # stream end (MiniMax buffered mode where we skip inline delta).
        pending_tool_calls: dict[int, dict[str, Any]] = {}

        async with self._global_rate_limiter.concurrency_slot():
            try:
                stream, body = await self._create_stream(body)

                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(),
                            timeout=chunk_timeout,
                        )
                    except TimeoutError:
                        logger.warning(
                            "{}_STREAM: keepalive heartbeat (timeout={}s)",
                            tag,
                            chunk_timeout,
                        )
                        try:
                            yield ": keepalive\n\n"
                        except Exception:
                            logger.warning(
                                "{}_STREAM: client disconnected during keepalive",
                                tag,
                            )
                            break
                        continue
    
                    except StopAsyncIteration:
                        break
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    logger.debug("RAW_DELTA model={} delta={}", model_str, delta)

                    if delta is None:
                        continue

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.debug("{} finish_reason: {}", tag, finish_reason)

                    # ---------------------------------------------------
                    # Flush pending tool calls immediately once arguments
                    # become valid JSON instead of waiting for finish_reason
                    # ---------------------------------------------------
                    if (
                        pending_tool_calls
                        and not quirks.buffer_tool_calls
                        and finish_reason
                    ):
                        completed_indices = []
                        for tc_index, tc_info in pending_tool_calls.items():
                            arguments = (
                                tc_info.get("function", {}).get("arguments", "").strip()
                            )

                            if not arguments:
                                continue

                            try:
                                json.loads(arguments)
                            except Exception:
                                continue

                            completed_indices.append(tc_index)

                        if completed_indices:
                            for event in sse.close_content_blocks():
                                yield event

                        for tc_index in completed_indices:
                            tc_info = pending_tool_calls.pop(tc_index)

                            repaired = self._validate_and_repair_pending_tool(
                                tc_info,
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

                            for event in self._process_tool_call(repaired, sse):
                                yield event

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
                    content = getattr(delta, "content", None)
                    if content:
                        for part in think_parser.feed(content):
                            if part.type == ContentType.THINKING:
                                if not thinking_enabled:
                                    continue
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                if part.content and part.content.strip():
                                    if pending_tool_calls:
                                        continue

                                    for event in sse.ensure_text_block():
                                        yield event

                                    yield sse.emit_text_delta(part.content)

                    # ---------------------------------------------------
                    # Native tool calls – accumulate chunks
                    # ---------------------------------------------------
                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_index = getattr(tc, "index", 0)
                            existing = pending_tool_calls.setdefault(
                                tc_index,
                                {
                                    "index": tc_index,
                                    "id": None,
                                    "function": {"name": "", "arguments": ""},
                                },
                            )

                            tc_id = getattr(tc, "id", None)
                            if tc_id:
                                existing["id"] = tc_id

                            tc_function = getattr(tc, "function", None)
                            if tc_function:
                                fn_name = getattr(tc_function, "name", None)
                                fn_args = getattr(tc_function, "arguments", None)
                                if fn_name:
                                    existing["function"]["name"] = fn_name
                                if fn_args:
                                    current_arguments = existing["function"][
                                        "arguments"
                                    ]

                                    if not current_arguments:
                                        existing["function"]["arguments"] = fn_args
                                    else:
                                        candidate_replace = fn_args.strip()
                                        candidate_append = current_arguments + fn_args

                                        replace_valid = False
                                        append_valid = False

                                        try:
                                            json.loads(candidate_replace)
                                            replace_valid = True
                                        except Exception:
                                            pass

                                        try:
                                            json.loads(candidate_append)
                                            append_valid = True
                                        except Exception:
                                            pass

                                        if replace_valid:
                                            existing["function"][
                                                "arguments"
                                            ] = candidate_replace
                                        elif append_valid:
                                            existing["function"][
                                                "arguments"
                                            ] = candidate_append
                                        else:
                                            existing["function"][
                                                "arguments"
                                            ] = candidate_append

            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as e:
                logger.exception("STREAM_EXCEPTION")
                self._log_stream_transport_error(tag, req_tag, e)
                mapped_e = map_error(e, rate_limiter=self._global_rate_limiter)
                base_message = user_visible_message_for_mapped_provider_error(
                    mapped_e,
                    provider_name=tag,
                    read_timeout_s=self._config.http_read_timeout,
                )
                error_message = append_request_id(base_message, request_id)
                logger.info(
                    "{}_STREAM: Emitting SSE error event for {}{}",
                    tag,
                    type(e).__name__,
                    req_tag,
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

        # ==================================================================
        # Post-stream processing
        # ==================================================================

        # ------------------------------------------------------------------
        # Buffered path (MiniMax): process all accumulated tool calls at once.
        #
        # FIX: Original order was:
        #   1. process pending_tool_calls (calls _process_tool_call → start_tool_block
        #      → emit_tool_delta which accumulates into raw_arg_buffer)
        #   2. pending_tool_calls.clear()   ← this happened BEFORE emit_buffered_tool_args
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
        #   3. clear pending_tool_calls
        # ------------------------------------------------------------------
        if quirks.buffer_tool_calls and pending_tool_calls:
            for event in sse.close_content_blocks():
                yield event

            for tc_info in list(pending_tool_calls.values()):
                repaired = self._validate_and_repair_pending_tool(tc_info, model_str)
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
                    tool_name = tc_info.get("function", {}).get("name", "tool")
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
                tc_index = repaired.get("index", 0)
                tc_id = repaired.get("id") or f"tool_{uuid.uuid4()}"
                tool_name = repaired.get("function", {}).get("name", "") or "tool_call"
                repaired_args = repaired.get("function", {}).get("arguments", "{}")

                # Register tool state if not already done during stream
                if (
                    tc_index not in sse.blocks.tool_states
                    or not sse.blocks.tool_states[tc_index].started
                ):
                    yield sse.start_tool_block(tc_index, str(tc_id), tool_name)

                # Store repaired args in raw_arg_buffer for emit_buffered_tool_args
                state = sse.blocks.tool_states.get(tc_index)
                if state is not None and not state.buffered_args_emitted:
                    # Overwrite raw_arg_buffer with the fully repaired JSON
                    # (replaces any partial chunks accumulated during stream)
                    state.raw_arg_buffer = repaired_args

            # FIX: emit_buffered_tool_args BEFORE clearing pending_tool_calls
            for event in sse.emit_buffered_tool_args(model_str):
                yield event

            pending_tool_calls.clear()

        # ------------------------------------------------------------------
        # Flush remaining think-parser content
        # ------------------------------------------------------------------
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if thinking_enabled:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(remaining.content)
            if remaining and remaining.type == ContentType.TEXT:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)
            # Final emergency flush for pending tool calls
        if pending_tool_calls:
            logger.warning(
                "FINAL_TOOL_FLUSH: emitting {} pending tool calls",
                len(pending_tool_calls),
            )
            for event in sse.close_content_blocks():
                yield event
            for tc_info in list(pending_tool_calls.values()):
                repaired = self._validate_and_repair_pending_tool(
                    tc_info,
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
                for event in self._process_tool_call(repaired, sse):
                    yield event
            pending_tool_calls.clear()
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

        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
