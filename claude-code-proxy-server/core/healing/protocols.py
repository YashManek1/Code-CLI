"""Core protocols for the healing architecture."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from .normalization import NormalizedEvent, ProviderProfile
from .taxonomy import FailureClassification

@runtime_checkable
class HealingEngine(Protocol):
    """Protocol for the semantic continuation engine."""
    def build_resumption_messages(
        self,
        original_messages: List[Dict[str, Any]],
        partial_content: str,
        partial_tool_calls: List[Dict[str, Any]],
        failure_context: str
    ) -> List[Dict[str, Any]]:
        ...

@runtime_checkable
class StreamValidator(Protocol):
    """Protocol for stream integrity verification."""
    def verify_chunk(self, chunk: Dict[str, Any], sequence_id: int) -> None:
        ...
    def get_stream_fingerprint(self) -> str:
        ...

@runtime_checkable
class RecoveryOrchestrator(Protocol):
    """Protocol for orchestrating the multi-stage recovery pipeline."""
    async def handle_failure(
        self,
        error: Exception,
        original_request: Dict[str, Any],
        partial_response: Optional[str] = None
    ) -> Dict[str, Any]:
        ...

@runtime_checkable
class TransactionManager(Protocol):
    """Protocol for transactional file editing and tool recovery."""
    def stage_edit(self, filepath: str, content: str) -> str: # returns staging path
        ...
    def validate_staging(self, staging_path: str) -> bool:
        ...
    def commit_edit(self, staging_path: str, target_path: str) -> None:
        ...
    def rollback_all(self) -> None:
        ...
