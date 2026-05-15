import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from api.app import create_app
from providers.openai_compat import OpenAIChatTransport
from api.models.anthropic import MessagesRequest
from config.settings import Settings
from providers.registry import ProviderRegistry
from datetime import datetime, UTC
import asyncio

class TestTransport(OpenAIChatTransport):
    """Concrete implementation of OpenAIChatTransport for testing."""
    def _build_request_body(self, request, thinking_enabled):
        return {"model": request.model, "messages": []}

@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.healing_enabled = True
    settings.max_healing_attempts = 3
    settings.validate_stream_integrity = True
    settings.log_raw_sse_events = False
    settings.log_api_error_tracebacks = True
    settings.provider_stream_idle_timeout = 45.0
    settings.enable_execution_state_orchestration = False
    settings.host = "0.0.0.0"
    settings.port = 8082
    settings.anthropic_auth_token = ""
    settings.auto_disable_thinking_simple_prompts = True
    settings.enable_weak_model_quality_hints = True
    settings.enable_model_thinking = True
    return settings

@pytest.mark.asyncio
async def test_proxy_pipeline_comprehensive(mock_settings):
    """Run both success and healing cases in a single test to ensure isolation is not an issue."""
    
    app = create_app(lifespan_enabled=False)
    app.state.provider_registry = MagicMock(spec=ProviderRegistry)
    
    from api.dependencies import get_settings
    app.dependency_overrides[get_settings] = lambda: mock_settings
    
    client = TestClient(app)
    
    # CASE 1: SUCCESS
    mock_provider = MagicMock()
    async def mock_stream_gen(*args, **kwargs):
        yield 'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        yield 'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"SUCCESS"}}\n\n'
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    mock_provider.stream_response.side_effect = mock_stream_gen
    mock_provider.preflight_stream = AsyncMock(return_value=None)
    
    app.state.provider_registry.get.return_value = mock_provider
    
    with patch("api.dependencies.resolve_provider", return_value=mock_provider), \
         patch("config.settings.Settings.resolve_model", return_value="test/test"), \
         patch("config.settings.Settings.resolve_thinking", return_value=True):
        
        response = client.post("/v1/messages", json={
            "model": "test/test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        })
        assert response.status_code == 200
        assert "SUCCESS" in response.text
        
    # CASE 2: HEALING
    transport_config = MagicMock()
    transport_config.proxy = None
    transport_config.log_raw_sse_events = False
    transport_config.http_read_timeout = 30.0
    transport_config.http_connect_timeout = 5.0
    transport_config.http_write_timeout = 5.0
    transport_config.rate_limit = 100
    transport_config.rate_window = 60
    transport_config.max_concurrency = 10
    transport_config.execution_state_store = None
    
    transport = TestTransport(
        transport_config,
        provider_name="test-healing",
        base_url="https://api.test.com",
        api_key="test_key"
    )
    
    async def impl_1(*args, **kwargs):
        yield 'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_2","model":"test-healing","usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        raise ConnectionError("Drop")
    async def impl_2(*args, **kwargs):
        yield 'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"HEALED"}}\n\n'
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            
    with patch.object(transport, "_stream_response_impl") as mock_impl:
        mock_impl.side_effect = [impl_1(), impl_2()]
        transport.preflight_stream = AsyncMock(return_value=None)
        
        app.state.provider_registry.get.return_value = transport
        
        with patch("api.dependencies.resolve_provider", return_value=transport), \
             patch("config.settings.Settings.resolve_model", return_value="test/test"), \
             patch("config.settings.Settings.resolve_thinking", return_value=True):
            
            response = client.post("/v1/messages", json={
                "model": "test/test",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "execution_state": None
            })
            
            assert response.status_code == 200
            assert "HEALED" in response.text
            assert mock_impl.call_count == 2
