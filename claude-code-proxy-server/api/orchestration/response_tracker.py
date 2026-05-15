"""Track tool responses and automatically update execution state.

Parses incoming user messages for tool results and updates the
execution state's plan steps if success signals are detected.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..execution_state_store import ExecutionStateStore
from .execution_tracker import ExecutionTracker


class ResponseTracker:
    """Monitors tool results to automatically advance PlanSteps."""

    def __init__(self, store: ExecutionStateStore) -> None:
        self._store = store
        self._tracker = ExecutionTracker(store)

    def process_request_messages(self, session_id: str, messages: list[Any]) -> None:
        """Scan incoming messages for tool results and update steps."""
        if not messages:
            return

        state = self._store.load(session_id)
        if state is None or not state.remaining_steps:
            return

        # Look for tool results in the latest user message
        last_msg = messages[-1]
        if self._get_field(last_msg, "role") != "user":
            return

        content = self._get_field(last_msg, "content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if self._get_field(block, "type") == "tool_result":
                tool_use_id = self._get_field(block, "tool_use_id")
                tool_use_id_string = str(tool_use_id) if tool_use_id else ""
                is_error = self._get_field(block, "is_error", False)

                if not tool_use_id_string or is_error:
                    continue

                if tool_use_id_string in state.processed_tool_result_ids:
                    logger.debug(
                        "AUTO_TRACK: duplicate tool_result ignored session={} tool_use_id={}",
                        session_id,
                        tool_use_id_string,
                    )
                    continue

                # Very basic auto-tracker: mark the active step as in_progress
                # if a tool result successfully returns. In a full implementation,
                # we would map the tool_use_id back to the assistant's action.
                active_step = self._tracker.get_next_step(session_id)
                if active_step and active_step.status == "pending":
                    logger.info(
                        "AUTO_TRACK: marking step_id={} as completed due to successful tool_result",
                        active_step.step_id,
                    )
                    self._tracker.mark_step_completed(session_id, active_step.step_id)

                latest = self._store.load(session_id)
                if latest is None:
                    return
                if tool_use_id_string not in latest.processed_tool_result_ids:
                    latest.processed_tool_result_ids.append(tool_use_id_string)
                    self._store.save(latest)
                state = latest

    @staticmethod
    def _get_field(value: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(field_name, default)
        return getattr(value, field_name, default)
