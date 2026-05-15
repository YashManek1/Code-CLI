import pytest
from core.healing.normalization import ProviderNormalizationLayer, NormalizedEvent, ProviderProfile

@pytest.fixture
def openai_profile():
    return ProviderProfile(name="openai-gpt-4", has_reasoning=True)

@pytest.fixture
def anthropic_profile():
    return ProviderProfile(name="anthropic-claude-3", preferred_format="anthropic")

def test_openai_normalization(openai_profile):
    norm = ProviderNormalizationLayer()
    chunk = {
        "choices": [{
            "delta": {
                "content": "hello",
                "reasoning_content": "thinking...",
                "tool_calls": [{"index": 0, "id": "tc1", "function": {"name": "search", "arguments": "{}"}}]
            },
            "finish_reason": "stop"
        }]
    }
    
    events = norm.normalize_chunk(chunk, openai_profile)
    assert any(e.type == "text" and e.content == "hello" for e in events)
    assert any(e.type == "reasoning" and e.content == "thinking..." for e in events)
    assert any(e.type == "tool_call" and e.tool_call_id == "tc1" for e in events)
    assert any(e.type == "stop" and e.finish_reason == "stop" for e in events)

def test_anthropic_normalization(anthropic_profile):
    norm = ProviderNormalizationLayer()
    
    # Text delta
    events = norm.normalize_chunk({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "world"}
    }, anthropic_profile)
    assert events[0].type == "text"
    assert events[0].content == "world"
    
    # Tool start
    events = norm.normalize_chunk({
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "t1", "name": "ls"}
    }, anthropic_profile)
    assert events[0].type == "tool_call"
    assert events[0].tool_call_id == "t1"
    assert events[0].tool_name == "ls"
