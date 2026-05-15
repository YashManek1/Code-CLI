import pytest
from core.healing.snapshots import SnapshotManager
from api.models.execution_state import ExecutionState

def test_snapshot_lifecycle():
    manager = SnapshotManager()
    state = ExecutionState(session_id="test_sess", implementation_phase="idle")
    
    # 1. Take snapshot
    sid = manager.take_snapshot(state, reason="initial")
    assert sid.startswith("snap_")
    assert len(manager._lineage) == 1
    
    # 2. Modify state
    state.implementation_phase = "backend_planning"
    state.active_files = ["file1.py"]
    
    # 3. Take another
    sid2 = manager.take_snapshot(state, reason="planning")
    assert len(manager._lineage) == 2
    
    # 4. Rollback
    restored = manager.rollback_to(sid)
    assert restored.implementation_phase == "idle"
    assert not restored.active_files
    assert len(manager._lineage) == 1

def test_snapshot_pruning():
    manager = SnapshotManager()
    state = ExecutionState(session_id="test", implementation_phase="idle")
    
    # Take 60 snapshots (limit is 50)
    for i in range(60):
        manager.take_snapshot(state, reason=f"snap_{i}")
        
    assert len(manager._lineage) == 50
    assert len(manager._snapshots) == 50
