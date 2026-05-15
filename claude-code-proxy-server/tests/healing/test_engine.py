import pytest
from core.healing.engine import HealingContinuationEngine

def test_continuation_prompt_construction():
    engine = HealingContinuationEngine()
    original = [{"role": "user", "content": "Write a function"}]
    partial_text = "def hello():"
    partial_tools = []
    
    new_msgs = engine.build_resumption_messages(original, partial_text, partial_tools, "TIMEOUT")
    
    # original(1) + assistant(1) + resumption(1) = 3
    assert len(new_msgs) == 3
    assert new_msgs[1]["role"] == "assistant"
    # content is a list of blocks
    assert new_msgs[1]["content"][0]["type"] == "text"
    assert "def hello():" in new_msgs[1]["content"][0]["text"]
    
    assert new_msgs[2]["role"] == "user"
    assert "SYSTEM_RESUMPTION" in new_msgs[2]["content"]

def test_tool_call_resumption():
    engine = HealingContinuationEngine()
    original = [{"role": "user", "content": "List files"}]
    partial_text = ""
    partial_tools = [{
        "id": "t1",
        "function": {"name": "ls", "arguments": '{"path": "."}'}
    }]
    
    new_msgs = engine.build_resumption_messages(original, partial_text, partial_tools, "CRASH")
    
    assert new_msgs[1]["content"][0]["type"] == "tool_use"
    assert new_msgs[1]["content"][0]["name"] == "ls"
    assert new_msgs[1]["content"][0]["input"] == {"path": "."}
