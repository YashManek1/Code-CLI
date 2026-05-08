import asyncio

import pytest

from api.models.anthropic import MessagesRequest
from api.response_cache import clear_response_cache_for_testing, dedupe_and_cache_stream


def _request(text: str, *, temperature: float | None = 0) -> MessagesRequest:
    return MessagesRequest(
        model="nvidia_nim/test-model",
        messages=[{"role": "user", "content": text}],
        temperature=temperature,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_response_cache_for_testing()
    yield
    clear_response_cache_for_testing()


@pytest.mark.asyncio
async def test_dedupe_replays_inflight_deterministic_stream():
    calls = 0
    first_event_started = asyncio.Event()

    async def factory():
        nonlocal calls
        calls += 1
        yield "event: message_start\ndata: {}\n\n"
        first_event_started.set()
        await asyncio.sleep(0.05)
        yield "event: message_stop\ndata: {}\n\n"

    request = _request("explain caches")

    async def consume(request_id: str):
        return [
            event
            async for event in dedupe_and_cache_stream(
                request,
                provider_id="nvidia_nim",
                request_id=request_id,
                factory=factory,
            )
        ]

    first = asyncio.create_task(consume("req_first"))
    await first_event_started.wait()
    second = asyncio.create_task(consume("req_second"))

    first_events, second_events = await asyncio.gather(first, second)

    assert calls == 1
    assert first_events == second_events


@pytest.mark.asyncio
async def test_cache_serves_completed_deterministic_stream():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    request = _request("explain caches")

    first_events = [
        event
        async for event in dedupe_and_cache_stream(
            request,
            provider_id="nvidia_nim",
            request_id="req_first",
            factory=factory,
        )
    ]
    second_events = [
        event
        async for event in dedupe_and_cache_stream(
            request,
            provider_id="nvidia_nim",
            request_id="req_second",
            factory=factory,
        )
    ]

    assert calls == 1
    assert first_events == second_events


@pytest.mark.asyncio
async def test_non_deterministic_requests_skip_cache():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        yield f"event: message_stop\ndata: {calls}\n\n"

    request = _request("write code", temperature=0.2)

    first_events = [
        event
        async for event in dedupe_and_cache_stream(
            request,
            provider_id="nvidia_nim",
            request_id="req_first",
            factory=factory,
        )
    ]
    second_events = [
        event
        async for event in dedupe_and_cache_stream(
            request,
            provider_id="nvidia_nim",
            request_id="req_second",
            factory=factory,
        )
    ]

    assert calls == 2
    assert first_events != second_events
