"""Detection for context poisoning and recursive repair hallucinations."""

from __future__ import annotations
from typing import List, Dict, Any
from loguru import logger

class ContextPoisonDetector:
    """Analyzes history to detect contradictions or recursive repair artifacts."""

    def __init__(self):
        self._max_repairs = 3

    def is_poisoned(self, retry_history: List[Dict[str, Any]], validation_failures: List[Dict[str, Any]]) -> bool:
        """Check if the current context is likely poisoned by recursive repairs."""
        
        # 1. Check for excessive repeated repairs of the same file/error
        if len(retry_history) >= self._max_repairs:
            last_3_types = [r.get("failure_type") for r in retry_history[-3:]]
            if len(set(last_3_types)) == 1 and last_3_types[0] == "ast_validation_failed":
                logger.warning("POISON_DETECTOR: Detected recursive AST repair loop.")
                return True

        # 2. Check for contradictory validation findings
        # e.g. "Missing import X" followed by "Unused import X" then "Missing import X" again
        # This implementation would need semantic similarity checks, for now we use simple count
        if len(validation_failures) > 10:
            logger.warning("POISON_DETECTOR: Excessive validation failures indicating unstable state.")
            return True

        return False

    def get_cleanup_recommendation(self, retry_history: List[Dict[str, Any]]) -> str:
        """Suggest how to clean up a poisoned context."""
        return (
            "Recursive repair detected. Recommendation: Rollback to the last stable snapshot "
            "and switch to a higher-fidelity provider for the repair phase."
        )
