import pytest
import binascii
from core.healing.integrity import StreamIntegrityValidator, StreamIntegrityError

def test_sequence_integrity():
    validator = StreamIntegrityValidator()
    
    # Valid sequence
    validator.verify_chunk({"choices": [{"delta": {"content": "a"}}]}, 0)
    validator.verify_chunk({"choices": [{"delta": {"content": "b"}}]}, 1)
    
    # Duplicate sequence should fail
    with pytest.raises(StreamIntegrityError, match="Duplicate/Out-of-order"):
        validator.verify_chunk({"choices": [{"delta": {"content": "b"}}]}, 1)
        
    # Gap detection
    with pytest.raises(StreamIntegrityError, match="Missing sequence range"):
        validator.verify_chunk({"choices": [{"delta": {"content": "d"}}]}, 3)

def test_crc_verification():
    validator = StreamIntegrityValidator()
    
    validator.verify_chunk({"choices": [{"delta": {"content": "hello"}}]}, 0)
    expected_crc = binascii.crc32(b"hello") & 0xFFFFFFFF
    assert validator._content_crc == expected_crc
    
    validator.verify_chunk({"choices": [{"delta": {"content": " world"}}]}, 1)
    expected_crc = binascii.crc32(b" world", expected_crc) & 0xFFFFFFFF
    assert validator._content_crc == expected_crc

def test_tool_call_boundaries():
    validator = StreamIntegrityValidator()
    
    # Start tool call
    validator.verify_chunk({
        "choices": [{
            "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "test"}}]}
        }]
    }, 0)
    
    # Normal finish
    validator.verify_chunk({
        "choices": [{"finish_reason": "tool_calls", "delta": {}}]
    }, 1)
    
    assert not validator._open_tool_calls

def test_incomplete_tool_call_at_end():
    validator = StreamIntegrityValidator()
    
    # Start tool call
    validator.verify_chunk({
        "choices": [{
            "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "test"}}]}
        }]
    }, 0)
    
    # Finish with "stop" but tool call still open
    with pytest.raises(StreamIntegrityError, match="Incomplete tool call boundaries"):
        validator.verify_chunk({
            "choices": [{"finish_reason": "stop", "delta": {}}]
        }, 1)
