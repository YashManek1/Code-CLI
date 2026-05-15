"""Normalization layer for cross-provider semantic consistency."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass, field
from loguru import logger

@dataclass(frozen=True)
class ProviderProfile:
    """Capabilities and characteristics of a specific model/provider."""
    name: str
    has_reasoning: bool = False
    supports_tools: bool = True
    max_tokens: int = 4096
    context_window: int = 128000
    preferred_format: str = "openai" # 'openai', 'anthropic'
    ast_correction_score: float = 0.5 # 0.0 to 1.0 (historical)

@dataclass
class NormalizedEvent:
    """Standardized representation of an LLM generation event."""
    type: str  # 'text', 'reasoning', 'tool_call', 'stop', 'error'
    content: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProviderNormalizationLayer:
    """Standardizes disparate provider outputs before they enter the healing system."""

    def normalize_chunk(self, chunk: Dict[str, Any], provider_profile: ProviderProfile) -> List[NormalizedEvent]:
        """Convert a provider-specific chunk into a list of normalized events."""
        provider_name = provider_profile.name.lower()
        
        if "anthropic" in provider_name:
            return self._normalize_anthropic(chunk)
        elif "openai" in provider_name or "nvidia_nim" in provider_name or "open_router" in provider_name:
            return self._normalize_openai(chunk)
        elif "google" in provider_name or "gemini" in provider_name:
            return self._normalize_gemini(chunk)
        
        # Fallback to OpenAI-like as it's the industry standard
        return self._normalize_openai(chunk)

    def _normalize_openai(self, chunk: Dict[str, Any]) -> List[NormalizedEvent]:
        events = []
        choices = chunk.get("choices", [])
        if not choices:
            # Handle potential non-choice chunks (e.g. usage info)
            return events

        delta = choices[0].get("delta", {})
        
        # 1. Text/Reasoning
        if content := delta.get("content"):
            events.append(NormalizedEvent(type="text", content=content))
        
        # Some providers use 'reasoning' or 'thought' fields
        if reasoning := (delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thought")):
            events.append(NormalizedEvent(type="reasoning", content=reasoning))

        # 2. Tool Calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            events.append(NormalizedEvent(
                type="tool_call",
                tool_call_index=tc.get("index"),
                tool_call_id=tc.get("id"),
                tool_name=tc.get("function", {}).get("name"),
                tool_input=tc.get("function", {}).get("arguments")
            ))

        # 3. Stop
        if finish_reason := choices[0].get("finish_reason"):
            events.append(NormalizedEvent(
                type="stop", 
                finish_reason=finish_reason
            ))

        return events

    def _normalize_anthropic(self, chunk: Dict[str, Any]) -> List[NormalizedEvent]:
        events = []
        msg_type = chunk.get("type")
        
        if msg_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                events.append(NormalizedEvent(type="text", content=delta["text"]))
            elif delta.get("type") == "thinking_delta": # For future Claude thinking models
                events.append(NormalizedEvent(type="reasoning", content=delta["thinking"]))
            elif delta.get("type") == "input_json_delta":
                events.append(NormalizedEvent(
                    type="tool_call",
                    tool_call_index=chunk.get("index"),
                    tool_input=delta["partial_json"]
                ))
        elif msg_type == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                events.append(NormalizedEvent(
                    type="tool_call",
                    tool_call_index=chunk.get("index"),
                    tool_call_id=block.get("id"),
                    tool_name=block.get("name")
                ))
        elif msg_type == "message_delta":
            delta = chunk.get("delta", {})
            if stop_reason := delta.get("stop_reason"):
                events.append(NormalizedEvent(
                    type="stop",
                    finish_reason=stop_reason
                ))

        return events

    def _normalize_gemini(self, chunk: Dict[str, Any]) -> List[NormalizedEvent]:
        # Google Gemini usually comes via an OpenAI-compatible proxy in this repo,
        # but if it's raw Vertex/AI Studio, we handle it here.
        # For now, it delegates to OpenAI.
        return self._normalize_openai(chunk)
