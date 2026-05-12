"""In-memory request dedupe and deterministic SSE response cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from api.models.anthropic import MessagesRequest

StreamFactory = Callable[[], AsyncIterator[str]]

_CACHE_MAX_ENTRIES = 500
_CODE_TTL_S = 5 * 60
_DOC_TTL_S = 60 * 60
_DUPLICATE_REPLAY_TTL_S = 30
_REPLAY_MAX_BYTES = 2 * 1024 * 1024
_STREAM_KEEPALIVE_INTERVAL_S = 10.0
_STREAM_KEEPALIVE_EVENT = ": keep-alive\n\n"


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    events: tuple[str, ...]


@dataclass(slots=True)
class _InflightEntry:
    done: asyncio.Event
    condition: asyncio.Condition
    events: list[str]
    error: BaseException | None = None
    producer_task: asyncio.Task[None] | None = None
    consumer_count: int = 0


_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()
_INFLIGHT: dict[str, _InflightEntry] = {}


def _request_payload(request: MessagesRequest, *, provider_id: str) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "request": request.model_dump(mode="json", exclude_none=True),
    }


def request_cache_key(request: MessagesRequest, *, provider_id: str) -> str:
    payload = _request_payload(request, provider_id=provider_id)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _iter_message_text(request: MessagesRequest) -> str:
    parts: list[str] = []
    for message in request.messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    system = request.system
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(block.text for block in system if block.text)
    return "\n".join(parts).lower()


def _cache_ttl_s(request: MessagesRequest) -> int:
    text = _iter_message_text(request)
    doc_words = (
        "explain",
        "documentation",
        "docs",
        "summarize",
        "describe",
        "what is",
        "how does",
    )
    return _DOC_TTL_S if any(word in text for word in doc_words) else _CODE_TTL_S


def is_cacheable_request(request: MessagesRequest) -> bool:
    temperature = 1.0 if request.temperature is None else request.temperature
    return temperature == 0 and not request.tools


def _completed_replay_ttl_s(request: MessagesRequest, *, cacheable: bool) -> int | None:
    """Return a completed-response replay TTL, or None when replay is unsafe.

    Tool-capable requests are only deduped while the provider call is in flight.
    Replaying a completed tool-bearing stream can leak a stale tool decision into
    a later turn, even when the request body is byte-for-byte identical.
    """
    if request.tools:
        return None
    if cacheable:
        return _cache_ttl_s(request)
    return _DUPLICATE_REPLAY_TTL_S


def _get_cached_events(key: str) -> tuple[str, ...] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry.expires_at <= time.monotonic():
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return entry.events


def _put_cached_events(key: str, events: tuple[str, ...], ttl_s: int) -> None:
    _CACHE[key] = _CacheEntry(expires_at=time.monotonic() + ttl_s, events=events)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX_ENTRIES:
        _CACHE.popitem(last=False)


def _events_are_replayable(events: tuple[str, ...]) -> bool:
    if not events:
        return False
    total_bytes = 0
    saw_message_stop = False
    for event in events:
        total_bytes += len(event.encode("utf-8", errors="ignore"))
        if total_bytes > _REPLAY_MAX_BYTES:
            return False
        lowered = event.lower()
        if "event: error" in lowered or '"type":"error"' in lowered:
            return False
        if "event: message_stop" in lowered or '"type":"message_stop"' in lowered:
            saw_message_stop = True
    return saw_message_stop


async def dedupe_and_cache_stream(
    request: MessagesRequest,
    *,
    provider_id: str,
    request_id: str,
    factory: StreamFactory,
) -> AsyncIterator[str]:
    """Yield an SSE stream while deduping identical in-flight calls.

    The first caller starts the producer while all matching callers consume the
    same live event buffer as it grows, avoiding duplicate upstream provider
    requests during client-side request storms. Completed replay remains
    restricted to no-tool requests: deterministic prompts get the normal cache
    TTL, non-deterministic prompts get a short duplicate-storm TTL, and
    tool-bearing requests use in-flight dedupe only.
    """

    cacheable = is_cacheable_request(request)
    key = request_cache_key(request, provider_id=provider_id)

    if (cached := _get_cached_events(key)) is not None:
        logger.info(
            "RESPONSE_CACHE: hit request_id={} cacheable={}",
            request_id,
            cacheable,
        )
        for event in cached:
            yield event
        return

    inflight = _INFLIGHT.get(key)
    if inflight is not None:
        logger.info(
            "REQUEST_DEDUPE: joined request_id={} cacheable={}",
            request_id,
            cacheable,
        )
    else:
        inflight = _InflightEntry(
            done=asyncio.Event(),
            condition=asyncio.Condition(),
            events=[],
        )
        _INFLIGHT[key] = inflight
        inflight.producer_task = asyncio.create_task(
            _produce_inflight_stream(
                inflight,
                request,
                provider_id=provider_id,
                request_id=request_id,
                cacheable=cacheable,
                key=key,
                factory=factory,
            )
        )

    inflight.consumer_count += 1
    cursor = 0
    try:
        while True:
            send_keepalive = False
            async with inflight.condition:
                while cursor >= len(inflight.events) and not inflight.done.is_set():
                    try:
                        await asyncio.wait_for(
                            inflight.condition.wait(),
                            timeout=_STREAM_KEEPALIVE_INTERVAL_S,
                        )
                    except TimeoutError:
                        send_keepalive = True
                        break
                if send_keepalive:
                    pending_events = ()
                    producer_done = False
                    producer_error = None
                else:
                    pending_events = tuple(inflight.events[cursor:])
                    cursor += len(pending_events)
                    producer_done = inflight.done.is_set()
                    producer_error = inflight.error

            if send_keepalive:
                yield _STREAM_KEEPALIVE_EVENT
                continue
            for event in pending_events:
                yield event

            if producer_done:
                if producer_error is not None:
                    raise producer_error
                return
    finally:
        inflight.consumer_count -= 1
        if (
            inflight.consumer_count <= 0
            and not inflight.done.is_set()
            and inflight.producer_task is not None
        ):
            inflight.producer_task.cancel()


async def _produce_inflight_stream(
    inflight: _InflightEntry,
    request: MessagesRequest,
    *,
    provider_id: str,
    request_id: str,
    cacheable: bool,
    key: str,
    factory: StreamFactory,
) -> None:
    try:
        async for event in factory():
            async with inflight.condition:
                inflight.events.append(event)
                inflight.condition.notify_all()
    except BaseException as exc:
        inflight.error = exc
    else:
        events = tuple(inflight.events)
        ttl_s = _completed_replay_ttl_s(request, cacheable=cacheable)
        if ttl_s is not None and _events_are_replayable(events):
            _put_cached_events(key, events, ttl_s)
            logger.info(
                "RESPONSE_CACHE: stored request_id={} events={} ttl_s={} cacheable={}",
                request_id,
                len(events),
                ttl_s,
                cacheable,
            )
    finally:
        async with inflight.condition:
            inflight.done.set()
            inflight.condition.notify_all()
        _INFLIGHT.pop(key, None)


def clear_response_cache_for_testing() -> None:
    _CACHE.clear()
    _INFLIGHT.clear()
