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


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    events: tuple[str, ...]


@dataclass(slots=True)
class _InflightEntry:
    done: asyncio.Event
    events: list[str]
    error: BaseException | None = None


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


async def dedupe_and_cache_stream(
    request: MessagesRequest,
    *,
    provider_id: str,
    request_id: str,
    factory: StreamFactory,
) -> AsyncIterator[str]:
    """Yield an SSE stream while deduping identical in-flight deterministic calls.

    The first caller streams live while recording the response. Concurrent callers
    for the same deterministic request await the same collection task and replay
    its events, avoiding a duplicate upstream provider request.
    """

    if not is_cacheable_request(request):
        async for event in factory():
            yield event
        return

    key = request_cache_key(request, provider_id=provider_id)
    cached = _get_cached_events(key)
    if cached is not None:
        logger.info("RESPONSE_CACHE: hit request_id={}", request_id)
        for event in cached:
            yield event
        return

    inflight = _INFLIGHT.get(key)
    if inflight is not None:
        logger.info("REQUEST_DEDUPE: joined request_id={}", request_id)
        await inflight.done.wait()
        if inflight.error is not None:
            raise inflight.error
        for event in inflight.events:
            yield event
        return

    inflight = _InflightEntry(done=asyncio.Event(), events=[])
    _INFLIGHT[key] = inflight
    try:
        async for event in factory():
            inflight.events.append(event)
            yield event
    except BaseException as exc:
        inflight.error = exc
        raise
    else:
        events = tuple(inflight.events)
        if events:
            _put_cached_events(key, events, _cache_ttl_s(request))
        logger.info(
            "RESPONSE_CACHE: stored request_id={} events={}", request_id, len(events)
        )
    finally:
        inflight.done.set()
        _INFLIGHT.pop(key, None)


def clear_response_cache_for_testing() -> None:
    _CACHE.clear()
    _INFLIGHT.clear()
