"""Track tool responses and automatically update execution state.

Parses incoming user messages for tool results and updates the
execution state's plan steps if success signals are detected.
"""

from __future__ import annotations

from loguru import logger

from ..execution_state_store import ExecutionStateStore
from .execution_tracker import ExecutionTracker


class ResponseTracker:
    """Monitors tool results to automatically advance PlanSteps."""

    def __init__(self, store: ExecutionStateStore) -> None:
        self._store = store
        self._tracker = ExecutionTracker(store)

    def process_request_messages(
        self, session_id: str, messages: list[dict]
    ) -> None:
        """Scan incoming messages for tool results and update steps."""
        if not messages:
            return

        state = self._store.load(session_id)
        if state is None or not state.remaining_steps:
            return

        # Look for tool results in the latest user message
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return

        content = last_msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                is_error = block.get("is_error", False)

                if not is_error and tool_use_id:
                    # Very basic auto-tracker: mark the active step as completed
                    # if a tool result successfully returns. In a full implementation,
                    # we would map the tool_use_id back to the assistant's action.
                    active_step = self._tracker.get_next_step(session_id)
                    if active_step:
                        logger.info(
                            "AUTO_TRACK: marking step_id={} as completed due to successful tool_result",
                            active_step.step_id,
                        )
                        self._tracker.mark_step_completed(session_id, active_step.step_id)
