"""Middleware for managing resilient streaming and checkpointing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from core.anthropic.sse import SSEBuilder

from .checkpoint import StreamCheckpoint


class HealingStreamManager:
    """Wraps an AsyncIterator[str] to track generation state and handle failures."""

    def __init__(self, session_id: str, request_id: str | None = None):
        self.session_id = session_id
        self.request_id = request_id
        self.checkpoint = StreamCheckpoint(last_chunk_at=time.monotonic())
        self._start_time = time.monotonic()

    async def wrap_stream(self, stream: AsyncIterator[str], sse_builder: SSEBuilder | None = None) -> AsyncIterator[str]:
        """Yield events from stream while updating the local checkpoint."""
        try:
            async for event in stream:
                if sse_builder:
                    self._update_checkpoint_from_sse(sse_builder)
                yield event
            self.checkpoint.is_complete = True
        except (asyncio.CancelledError, Exception) as exc:
            logger.warning(
                "STREAM_INTERRUPTED: session={} request={} exc_type={} content_len={}",
                self.session_id,
                self.request_id,
                type(exc).__name__,
                len(self.checkpoint.content),
            )
            # Re-raise so the recovery orchestrator can catch it
            raise exc

    def _update_checkpoint_from_sse(self, sse: SSEBuilder) -> None:
        """Update checkpoint from SSEBuilder state."""
        self.checkpoint.last_chunk_at = time.monotonic()
        self.checkpoint.content = sse.accumulated_text
        self.checkpoint.reasoning = sse.accumulated_reasoning
        self.checkpoint.tool_calls = sse.accumulated_tool_calls

    def get_resumption_state(self) -> dict:
        """Return state needed for semantic continuation."""
        return {
            "content": self.checkpoint.content,
            "reasoning": self.checkpoint.reasoning,
            "tool_calls": list(self.checkpoint.tool_calls.values()),
            "elapsed": time.monotonic() - self._start_time,
        }
