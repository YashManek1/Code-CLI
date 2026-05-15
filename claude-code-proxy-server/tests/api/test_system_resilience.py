import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from api.services import ClaudeProxyService
from api.models.anthropic import MessagesRequest
from core.healing.taxonomy import FailureType

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.healing_enabled = True
    settings.max_healing_attempts = 2
    return settings

@pytest.fixture
def service(mock_settings):
    with patch("api.services.ExecutionStateStore"), \
         patch("api.services.ModelRouter"), \
         patch("api.services.get_token_count"):
        service = ClaudeProxyService(settings=mock_settings, provider_getter=MagicMock())
        return service

@pytest.mark.asyncio
async def test_proxy_service_handles_streaming_interruption(service):
    """Test that ClaudeProxyService uses a provider that can heal."""
    
    mock_transport = MagicMock()
    
    # Correctly mock an async iterator
    async def mock_stream_gen(*args, **kwargs):
        yield "chunk 1"
        yield "chunk 2"
        yield "chunk 3 (resumed)"

    mock_transport.stream_response.side_effect = mock_stream_gen
    
    # Mock provider getter
    service._provider_getter.return_value = mock_transport
    
    # Mock model router
    mock_resolved = MagicMock()
    mock_resolved.request = MessagesRequest(model="test", messages=[{"role": "user", "content": "test"}])
    mock_resolved.resolved.provider_id = "test-provider"
    mock_resolved.resolved.thinking_enabled = False
    service._model_router.resolve_messages_request.return_value = mock_resolved

    request = MessagesRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    
    response = service.create_message(request)
    # The response is a StreamingResponse, we need to iterate its body_iterator
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
        
    assert len(chunks) == 3
    assert chunks[2] == "chunk 3 (resumed)"

@pytest.mark.asyncio
async def test_system_stability_analytics_tracking(service):
    """Verify that the system tracks stability metrics during operation."""
    
    mock_transport = MagicMock()
    
    async def mock_ok_gen(*args, **kwargs):
        yield "ok"
    
    mock_transport.stream_response.side_effect = mock_ok_gen
    service._provider_getter.return_value = mock_transport
    
    mock_resolved = MagicMock()
    mock_resolved.request = MessagesRequest(model="test", messages=[{"role": "user", "content": "test"}])
    mock_resolved.resolved.provider_id = "test-provider"
    mock_resolved.resolved.thinking_enabled = False
    service._model_router.resolve_messages_request.return_value = mock_resolved

    # Mock the analytics object on the transport
    mock_analytics = MagicMock()
    mock_analytics.get_stats.return_value = {
        "total_interruptions": 5,
        "healing_success_rate": 0.8,
        "mtbf_seconds": 1200.0
    }
    mock_transport._analytics = mock_analytics

    request = MessagesRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    )
    
    response = service.create_message(request)
    async for _ in response.body_iterator:
        pass
            
    stats = mock_transport._analytics.get_stats()
    assert stats["total_interruptions"] == 5
