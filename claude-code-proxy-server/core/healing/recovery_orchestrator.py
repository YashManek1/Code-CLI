"""Logic for determining recovery strategies and semantic continuation."""

from __future__ import annotations

from typing import Any
from loguru import logger

from api.models.execution_state import ExecutionState


class RecoveryOrchestrator:
    """Decides how to recover from generation failures."""

    def __init__(self, state: ExecutionState):
        self.state = state

    def get_recovery_request(
        self, 
        original_messages: list[dict[str, Any]], 
        checkpoint_data: dict,
        error: Exception
    ) -> list[dict[str, Any]] | None:
        """Construct a new messages list to resume generation.
        
        If we have partial content, we append an 'assistant' message with that content
        and a 'user' message with a continuation instruction.
        """
        content = checkpoint_data.get("content", "")
        tool_calls = checkpoint_data.get("tool_calls", [])
        
        if not content and not tool_calls:
            logger.info("RECOVERY: No partial content to resume from. Retrying original.")
            return original_messages

        new_messages = list(original_messages)
        
        # Build the assistant's partial response
        assistant_content: list[dict[str, Any]] = []
        if content:
            assistant_content.append({"type": "text", "text": content})
        
        for tc in tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.get("id", "unknown"),
                "name": tc["function"]["name"],
                "input": self._parse_safe_json(tc["function"]["arguments"]),
            })

        if assistant_content:
            new_messages.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Add a resumption cue for the model
            new_messages.append({
                "role": "user",
                "content": (
                    "[RESUMPTION: The previous connection was interrupted. "
                    "I have preserved your output so far above. Please continue "
                    "exactly from where you left off to complete the task.]"
                )
            })
            
            logger.info("RECOVERY: Built resumption request with {} chars of partial content", len(content))
            
            # Update state with retry info
            if self.state:
                self.state.retry_history.append({
                    "timestamp": "now", # In a real app, use datetime
                    "error": str(error),
                    "partial_content_len": len(content)
                })
                
            return new_messages
            
        return original_messages

    def _parse_safe_json(self, raw: str) -> dict:
        import json
        try:
            return json.loads(raw)
        except Exception:
            # If JSON is incomplete, we could use repair_tool_arguments here
            return {}

    def get_failover_model(self, current_model: str) -> str | None:
        """Determine a failover model if the current one is repeatedly failing."""
        # Simple static failover for now
        failover_map = {
            "anthropic/claude-3.5-sonnet": "google/gemini-pro-1.5",
            "anthropic/claude-3-opus": "google/gemini-pro-1.5",
            "nvidia_nim/meta/llama-3.1-70b-instruct": "nvidia_nim/meta/llama-3.1-405b-instruct",
        }
        
        for key, val in failover_map.items():
            if key in current_model:
                logger.info("FAILOVER: Switching from {} to {}", current_model, val)
                return val
        
        return None
