import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from types import SimpleNamespace
from providers.openai_compat import OpenAIChatTransport
from core.healing.taxonomy import FailureType, FailureClassification
from core.healing.snapshots import ExecutionSnapshot

class MockChunk:
    def __init__(self, content, finish_reason=None):
        self.choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason
            )
        ]
        self.usage = None

class MinimalTransport(OpenAIChatTransport):
    def _build_request_body(self, request, thinking_enabled=None):
        return {"model": "test-model", "messages": []}
    
    async def _create_stream(self, body: dict):
        stream = await self._client.chat.completions.create(**body, stream=True)
        return stream, body

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.api_key = "test_key"
    config.base_url = "https://api.openai.com/v1"
    config.model = "gpt-4"
    config.proxy = None
    config.rate_limit = 100
    config.rate_window = 60
    config.max_concurrency = 5
    config.healing_enabled = True
    config.max_healing_attempts = 3
    config.chunk_timeout = 2.0
    config.validate_stream_integrity = True
    config.http_read_timeout = 10.0
    config.http_connect_timeout = 5.0
    config.http_write_timeout = 5.0
    config.log_api_error_tracebacks = False
    config.execution_state_store = None
    return config

@pytest.fixture
def transport(mock_config):
    with patch("openai.AsyncOpenAI"):
        transport = MinimalTransport(
            mock_config,
            provider_name="test",
            base_url="https://api.test.com",
            api_key="test_key"
        )
        # Mock the underlying HTTP client to avoid real network calls
        transport._client._client = MagicMock() 
        return transport

@pytest.mark.asyncio
async def test_resilience_from_mid_stream_transport_error(transport):
    """Test that the transport recovers from a mid-stream connection drop."""
    
    class AsyncIter:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    class AsyncIterFail:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items:
                raise ConnectionError("Stream dropped")
            return self.items.pop(0)

    async def mock_call_fail(*args, **kwargs):
        return AsyncIterFail([
            MockChunk("chunk 1 "),
            MockChunk("chunk 2 ")
        ])
        
    async def mock_call_success(*args, **kwargs):
        return AsyncIter([
            MockChunk("chunk 3 ", finish_reason="stop")
        ])
        
    def mock_create(*args, **kwargs):
        if transport._client.chat.completions.create.call_count == 1:
            return mock_call_fail()
        return mock_call_success()
        
    transport._client.chat.completions.create = MagicMock(side_effect=mock_create)

    # Mock continuation engine properly
    transport._continuation_engine = MagicMock()
    transport._continuation_engine.build_resumption_messages.return_value = [{"role": "user", "content": "Hello"}]

    from api.models.execution_state import ExecutionState
    transport._execution_state = ExecutionState(session_id="default")

    with patch("providers.openai_compat.ExecutionSnapshot") as mock_snap:
        mock_snap.return_value.snapshot_id = "test_snap_id"
        mock_snap.return_value.state = transport._execution_state
        
        chunks = []
        async for chunk in transport.stream_response([{"role": "user", "content": "Hello"}]):
            chunks.append(chunk)

        # Check that completions.create was called at least twice
        assert transport._client.chat.completions.create.call_count >= 2
        
        # Check that StabilityAnalytics tracked it
        stats = transport._analytics.get_kpis()
        assert stats.get("total_interruptions", 0) >= 1

@pytest.mark.asyncio
async def test_poison_detection_prevents_infinite_retries(transport):
    """Test that the poison detector aborts after repeated identical failures."""
    
    async def permanent_fail(*args, **kwargs):
        raise ConnectionError("Permanent Fail")
        if False: yield "" # Make it an async generator

    transport._client.chat.completions.create = AsyncMock(side_effect=permanent_fail)

    with patch("providers.openai_compat.ExecutionSnapshot") as mock_snap:
        mock_snap.return_value.snapshot_id = "test_snap_id"
        with pytest.raises(ConnectionError):
            async for _ in transport.stream_response([{"role": "user", "content": "Hello"}]):
                pass

    # Should have attempted retries but eventually given up
    assert transport._client.chat.completions.create.call_count > 1
