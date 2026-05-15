"""Deterministic execution journal for auditing and replay."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class JournalEntry(BaseModel):
    """A single recordable event in the execution timeline."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str # 'reasoning', 'tool_call', 'recovery', 'validation', 'rollback'
    provider: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExecutionJournal:
    """Audit log for all reasoning transitions and recovery events."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entries: List[JournalEntry] = []

    def record(self, event_type: str, provider: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an event in the journal."""
        entry = JournalEntry(
            event_type=event_type,
            provider=provider,
            content=content,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        
        # Optional: Persistent logging to disk for auditability
        # self._flush_to_disk(entry)

    def get_history(self) -> List[JournalEntry]:
        """Return the complete execution history."""
        return self.entries

    def export_json(self) -> str:
        """Export the journal as a JSON string."""
        return json.dumps([e.model_dump(mode='json') for e in self.entries], indent=2)

    def detect_loops(self) -> bool:
        """Detect repeated reasoning loops or identical tool calls."""
        if len(self.entries) < 4:
            return False
            
        # Basic check for repeated tool calls with identical inputs
        last_tool_calls = [e for e in self.entries if e.event_type == 'tool_call'][-3:]
        if len(last_tool_calls) == 3:
            contents = [json.dumps(tc.content) for tc in last_tool_calls]
            if contents[0] == contents[1] == contents[2]:
                return True
        return False
