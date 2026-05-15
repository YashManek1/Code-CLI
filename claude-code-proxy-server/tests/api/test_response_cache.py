import asyncio

import pytest

import api.response_cache as response_cache
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
async def test_dedupe_joiner_receives_live_events_before_primary_finishes():
    calls = 0
    first_event_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def factory():
        nonlocal calls
        calls += 1
        yield "event: message_start\ndata: {}\n\n"
        first_event_started.set()
        await release_stop.wait()
        yield "event: message_stop\ndata: {}\n\n"

    request = _request("explain caches")

    async def consume_all(request_id: str):
        return [
            event
            async for event in dedupe_and_cache_stream(
                request,
                provider_id="nvidia_nim",
                request_id=request_id,
                factory=factory,
            )
        ]

    async def consume_first_event(request_id: str):
        events = []
        async for event in dedupe_and_cache_stream(
            request,
            provider_id="nvidia_nim",
            request_id=request_id,
            factory=factory,
        ):
            events.append(event)
            if event.startswith("event: message_start"):
                break
        return events

    first = asyncio.create_task(consume_all("req_first"))
    await first_event_started.wait()
    second_events = await asyncio.wait_for(
        consume_first_event("req_second"), timeout=0.2
    )
    release_stop.set()
    first_events = await first

    assert calls == 1
    assert first_events == [
        "event: message_start\ndata: {}\n\n",
        "event: message_stop\ndata: {}\n\n",
    ]
    assert second_events == ["event: message_start\ndata: {}\n\n"]


@pytest.mark.asyncio
async def test_waiting_consumer_receives_keepalive_before_first_provider_event(
    monkeypatch,
):
    monkeypatch.setattr(response_cache, "_STREAM_KEEPALIVE_INTERVAL_S", 0.01)

    async def factory():
        await asyncio.sleep(0.05)
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    request = _request("explain caches")
    events = []
    async for event in dedupe_and_cache_stream(
        request,
        provider_id="nvidia_nim",
        request_id="req_keepalive",
        factory=factory,
    ):
        events.append(event)
        if event.startswith("event: message_start"):
            break

    assert events[0] == ": keep-alive\n\n"
    assert events[-1] == "event: message_start\ndata: {}\n\n"


@pytest.mark.asyncio
async def test_dedupe_replays_tool_request_only_while_inflight():
    calls = 0
    first_event_started = asyncio.Event()

    async def factory():
        nonlocal calls
        calls += 1
        yield "event: message_start\ndata: {}\n\n"
        first_event_started.set()
        await asyncio.sleep(0.05)
        yield f"event: message_stop\ndata: {calls}\n\n"

    request = MessagesRequest(
        model="nvidia_nim/test-model",
        messages=[{"role": "user", "content": "use a tool"}],
        temperature=0,
        tools=[
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object"},
            }
        ],
    )

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

    third_events = await consume("req_third")

    assert calls == 2
    assert third_events != first_events


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
async def test_non_deterministic_exact_duplicates_use_short_storm_replay():
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

    assert calls == 1
    assert first_events == second_events


@pytest.mark.asyncio
async def test_error_streams_are_not_replayed_after_completion():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        yield 'event: error\ndata: {"type":"error"}\n\n'

    request = _request("temporary provider failure", temperature=0.2)

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

    await consume("req_first")
    await consume("req_second")

    assert calls == 2
