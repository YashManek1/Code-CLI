"""Semantic continuation engine for intent-aligned healing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

class HealingContinuationEngine:
    """Reconstructs intent and builds continuation prompts after failures."""

    def build_resumption_messages(
        self, 
        original_messages: List[Dict[str, Any]], 
        partial_content: str,
        partial_tool_calls: List[Dict[str, Any]],
        failure_context: str
    ) -> List[Dict[str, Any]]:
        """Construct a semantically aligned continuation prompt."""
        
        new_messages = list(original_messages)
        
        # 1. Build the partial assistant response
        assistant_content: List[Dict[str, Any]] = []
        
        if partial_content:
            # Clean up trailing fragments to ensure clean continuation
            clean_content = self._trim_partial_content(partial_content)
            assistant_content.append({"type": "text", "text": clean_content})
            
        for tc in partial_tool_calls:
            # Handle cases where tool calls might be wrapped in a list (some provider quirks)
            if isinstance(tc, list) and len(tc) > 0:
                tc = tc[0]
                
            if not isinstance(tc, dict):
                logger.warning("CONTINUATION_ENGINE: Skipping non-dict tool call: {}", type(tc))
                continue

            assistant_content.append({
                "type": "tool_use",
                "id": tc.get("id", "resumed_tc"),
                "name": tc["function"]["name"],
                "input": self._parse_safe_json(tc["function"]["arguments"]),
            })

        if not assistant_content:
            logger.info("CONTINUATION_ENGINE: No partial content to resume. Returning original.")
            return original_messages

        # 2. Append the partial work
        new_messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        # 3. Add the Resumption Directive
        resumption_cue = self._generate_resumption_cue(partial_content, failure_context)
        new_messages.append({
            "role": "user",
            "content": resumption_cue
        })

        return new_messages

    def _trim_partial_content(self, content: str) -> str:
        """Trim incomplete words or fragments at the end of a stream."""
        # Simple heuristic: if the last char isn't punctuation or whitespace, 
        # it might be a fragment. But for coding, we often want to keep it.
        # For now, we keep it as is but could add smarter truncation here.
        return content

    def _generate_resumption_cue(self, partial_content: str, failure_context: str) -> str:
        """Build a context-aware directive for the model."""
        # Detect if we were mid-code block or mid-sentence
        is_mid_code = "```" in partial_content and partial_content.count("```") % 2 != 0
        
        last_fragment = partial_content[-100:].strip() if partial_content else "the start"
        
        cue = [
            f"[SYSTEM_RESUMPTION: Connection interrupted during {failure_context}. ",
            "I have preserved your progress above in the assistant message. ",
            f"Please continue exactly from where you left off (starting after: '{last_fragment}')."
        ]
        
        if is_mid_code:
            cue.append(" NOTE: You were in the middle of a code block. Do NOT repeat the opening ``` tags, just continue the code.")
        
        cue.append(" Do not repeat content already generated. Resuming now:]")
        
        return "".join(cue)

    def _parse_safe_json(self, raw: str) -> Dict[str, Any]:
        import json
        try:
            # Strip markdown blocks if present
            if raw.strip().startswith("```"):
                lines = raw.strip().split("\n")
                raw = "\n".join(lines[1:-1])
            return json.loads(raw)
        except Exception:
            # If JSON is corrupted, we should ideally use our repair_tool_arguments
            return {"_corrupted": True, "_raw": raw}

class HealingStrategyMatrix:
    """Maps failure classifications to specific recovery strategies."""
    
    def get_strategy(self, failure: FailureClassification) -> str:
        """Determine the best recovery strategy."""
        
        if failure.type == FailureType.CONNECTION_INTERRUPTED:
            return "semantic_continuation"
        
        if failure.type == FailureType.AST_VALIDATION_FAILED:
            return "repair_loop_with_context"
        
        if failure.severity == FailureSeverity.CRITICAL:
            return "escalate_to_human"
            
        if failure.type == FailureType.RATE_LIMIT_EXCEEDED:
            return "wait_and_retry"
            
        return "generic_retry"

        return "generic_retry"
