"""Cognitive versioning and snapshot management for execution states."""

from __future__ import annotations
import copy
from datetime import UTC, datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from api.models.execution_state import ExecutionState

class ExecutionSnapshot(BaseModel):
    """A point-in-time snapshot of the complete cognitive state."""
    snapshot_id: str
    parent_snapshot_id: Optional[str] = None
    state: ExecutionState
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str = "manual"
    semantic_score: float = 1.0

class SnapshotManager:
    """Manages the lifecycle of cognitive snapshots and rollbacks."""

    def __init__(self):
        self._snapshots: Dict[str, ExecutionSnapshot] = {}
        self._lineage: List[str] = []

    def take_snapshot(self, state: ExecutionState, reason: str) -> str:
        """Create a new snapshot of the current state."""
        import uuid
        snapshot_id = f"snap_{int(datetime.now(UTC).timestamp())}_{uuid.uuid4().hex[:8]}"
        parent_id = self._lineage[-1] if self._lineage else None
        
        # Deep copy the state to ensure isolation
        snapshot = ExecutionSnapshot(
            snapshot_id=snapshot_id,
            parent_snapshot_id=parent_id,
            state=copy.deepcopy(state),
            reason=reason
        )
        
        self._snapshots[snapshot_id] = snapshot
        self._lineage.append(snapshot_id)
        
        # Prune old snapshots if lineage is too long (Memory Management)
        if len(self._lineage) > 50:
            old_id = self._lineage.pop(0)
            self._snapshots.pop(old_id, None)
            
        return snapshot_id

    def rollback_to(self, snapshot_id: str) -> ExecutionState:
        """Restore state from a previous snapshot."""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self._snapshots[snapshot_id]
        # Update lineage to point to the restored snapshot
        try:
            idx = self._lineage.index(snapshot_id)
            self._lineage = self._lineage[:idx+1]
        except ValueError:
            self._lineage.append(snapshot_id)
            
        return copy.deepcopy(snapshot.state)

    def get_latest_snapshot(self) -> Optional[ExecutionSnapshot]:
        if not self._lineage:
            return None
        return self._snapshots.get(self._lineage[-1])

# Rebuild model to resolve forward references
ExecutionSnapshot.model_rebuild()
