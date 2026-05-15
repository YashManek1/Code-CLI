"""Data models for mid-stream generation checkpoints."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class StreamCheckpoint(BaseModel):
    """Captures the state of a streaming response for resumption."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, Any]] = Field(default_factory=dict)
    is_complete: bool = False
    last_chunk_at: float = Field(default_factory=float)
    
    def merge_chunk(self, content_delta: str = "", reasoning_delta: str = "", tool_call_delta: dict[str, Any] | None = None) -> None:
        """Merge a new chunk into the checkpoint."""
        if content_delta:
            self.content += content_delta
        if reasoning_delta:
            self.reasoning += reasoning_delta
        if tool_call_delta:
            idx = tool_call_delta.get("index", 0)
            if idx not in self.tool_calls:
                self.tool_calls[idx] = {"index": idx, "function": {"name": "", "arguments": ""}}
            
            tc = self.tool_calls[idx]
            func = tc["function"]
            
            new_func = tool_call_delta.get("function", {})
            if new_func.get("name"):
                func["name"] = new_func["name"]
            if new_func.get("arguments"):
                func["arguments"] += new_func["arguments"]
            
            if tool_call_delta.get("id"):
                tc["id"] = tool_call_delta["id"]

    def has_content(self) -> bool:
        """Return True if any content or reasoning has been accumulated."""
        return bool(self.content or self.reasoning or self.tool_calls)
