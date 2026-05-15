"""Stream integrity verification for high-fidelity healing."""

from __future__ import annotations

import binascii
from typing import Any, Dict, List, Optional
from loguru import logger

class StreamIntegrityError(Exception):
    """Raised when a stream fails integrity verification."""
    pass

class StreamIntegrityValidator:
    """Detects duplicated chunks, missing token ranges, and corrupted tool call boundaries."""

    def __init__(self):
        self._last_sequence_id: int = -1
        self._content_crc: int = 0
        self._open_tool_calls: Dict[int, str] = {} # index -> id

    def verify_chunk(self, chunk: Dict[str, Any], sequence_id: int) -> None:
        """Verify the integrity of an incoming SSE chunk.
        
        Args:
            chunk: The parsed SSE chunk.
            sequence_id: The sequential ID of the chunk in the stream.
        """
        # 1. Sequence Verification (Deterministic Ordering)
        if sequence_id <= self._last_sequence_id:
            logger.warning("STREAM_INTEGRITY: Duplicated or out-of-order chunk detected (seq={})", sequence_id)
            raise StreamIntegrityError(f"Duplicate/Out-of-order sequence: {sequence_id} <= {self._last_sequence_id}")
        
        # Detect gaps in sequence
        if self._last_sequence_id != -1 and sequence_id != self._last_sequence_id + 1:
            gap = sequence_id - self._last_sequence_id - 1
            logger.error("STREAM_INTEGRITY: Gap detected in stream sequence (gap_size={})", gap)
            raise StreamIntegrityError(f"Missing sequence range: {self._last_sequence_id + 1} to {sequence_id - 1}")

        self._last_sequence_id = sequence_id

        # 2. Delta CRC Update (High-Fidelity verification)
        content_delta = self._extract_content_delta(chunk)
        if content_delta:
            # Use CRC32 for fast, efficient integrity checking of the accumulated content
            self._content_crc = binascii.crc32(content_delta.encode('utf-8'), self._content_crc) & 0xFFFFFFFF

        # 3. Tool Call Boundary Verification
        self._track_tool_boundaries(chunk)

    def _extract_content_delta(self, chunk: Dict[str, Any]) -> str:
        """Extract text or reasoning delta from a standard OpenAI/Anthropic chunk."""
        # 1. OpenAI format
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content") or delta.get("reasoning_content") or ""
        
        # 2. Anthropic format
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            return delta.get("text") or delta.get("partial_json") or ""
            
        return ""

    def _track_tool_boundaries(self, chunk: Dict[str, Any]) -> None:
        """Track start/end of tool calls to detect corruption."""
        choices = chunk.get("choices", [])
        if not choices:
            return

        tool_calls = choices[0].get("delta", {}).get("tool_calls", [])
        for tc in tool_calls:
            index = tc.get("index", 0)
            tc_id = tc.get("id")
            
            if tc_id:
                # New tool call started
                self._open_tool_calls[index] = tc_id
            
        # Check for finish_reason
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "tool_calls":
            # All tool calls should have been closed/processed
            self._open_tool_calls.clear()
        elif finish_reason and self._open_tool_calls:
            logger.error("STREAM_INTEGRITY: Stream finished with unclosed tool calls: {}", self._open_tool_calls)
            raise StreamIntegrityError("Incomplete tool call boundaries at stream end")

    def get_stream_fingerprint(self) -> str:
        """Returns a hex fingerprint of the stream state."""
        return f"{self._content_crc:08x}"
