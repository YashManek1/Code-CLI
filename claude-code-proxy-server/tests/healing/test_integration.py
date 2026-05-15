import pytest
import os
import shutil
import tempfile
from core.healing.taxonomy import FailureTaxonomy, FailureType, FailureClassification, FailureSeverity
from core.healing.retry_controller import AdaptiveRetryController
from core.healing.router import CapabilityAwareRouter
from core.healing.normalization import ProviderProfile
from core.healing.engine import HealingContinuationEngine
from core.healing.snapshots import SnapshotManager
from core.healing.journal import ExecutionJournal
from core.healing.poison_detector import ContextPoisonDetector
from core.healing.transactions import TransactionalFileEditor
from core.healing.lifecycle import StabilityAnalytics
from api.models.execution_state import ExecutionState

@pytest.fixture
def workspace():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

def test_failure_classification():
    # Transport failure
    fail = FailureTaxonomy.classify(TimeoutError("request timed out"))
    assert fail.type == FailureType.PROVIDER_TIMEOUT
    assert fail.severity == FailureSeverity.MEDIUM
    
    # AST failure
    fail = FailureTaxonomy.classify(SyntaxError("invalid syntax"))
    assert fail.type == FailureType.AST_VALIDATION_FAILED
    assert fail.severity == FailureSeverity.MEDIUM

@pytest.mark.asyncio
async def test_adaptive_retry_logic():
    controller = AdaptiveRetryController()
    fail = FailureClassification(type=FailureType.PROVIDER_TIMEOUT, severity=FailureSeverity.MEDIUM, message="test")
    
    import unittest.mock as mock
    with mock.patch("asyncio.sleep", return_value=None) as mock_sleep:
        await controller.wait_before_retry("openai", fail)
        mock_sleep.assert_called_once()
        # Check exponential backoff
        await controller.wait_before_retry("openai", fail)
        assert mock_sleep.call_args_list[1][0][0] > mock_sleep.call_args_list[0][0][0]

def test_capability_aware_routing():
    p1 = ProviderProfile(name="fast-model", ast_correction_score=0.3)
    p2 = ProviderProfile(name="smart-model", ast_correction_score=0.9)
    router = CapabilityAwareRouter([p1, p2])
    
    fail = FailureClassification(
        type=FailureType.AST_VALIDATION_FAILED, 
        severity=FailureSeverity.MEDIUM, 
        message="syntax error"
    )
    
    # Should route away from failing fast-model to smart-model
    best = router.select_recovery_provider("fast-model", fail)
    assert best.name == "smart-model"

def test_semantic_continuation_mid_code():
    engine = HealingContinuationEngine()
    partial = "I will implement this:\n```python\ndef hello():\n    print('hi"
    
    messages = engine.build_resumption_messages(
        original_messages=[{"role": "user", "content": "write hello"}],
        partial_content=partial,
        partial_tool_calls=[],
        failure_context="streaming"
    )
    
    resumption_cue = messages[-1]["content"]
    assert "middle of a code block" in resumption_cue
    
    # Test poison detection (simulate 10 retries)
    state = ExecutionState(session_id="test")
    for _ in range(10):
        state.retry_history.append({"failure_type": "ast_validation_failed", "timestamp": "now"})
    
    detector = ContextPoisonDetector()
    assert detector.is_poisoned(state.retry_history, [])

def test_stability_analytics_kpis():
    analytics = StabilityAnalytics()
    analytics.record_event("stream_interruptions")
    analytics.record_event("stream_interruptions")
    analytics.record_event("successful_healing")
    analytics.record_event("hallucination")
    
    kpis = analytics.get_kpis()
    assert kpis["total_interruptions"] == 2
    assert kpis["hallucination_rate"] == 0.5
    assert "mtbf_seconds" in kpis

def test_snapshot_and_rollback():
    manager = SnapshotManager()
    state = ExecutionState(session_id="test", implementation_phase="idle")
    manager.take_snapshot(state, reason="baseline")
    
    state.implementation_phase = "backend_planning"
    restored = manager.rollback_to(manager._lineage[0])
    assert restored.implementation_phase == "idle"

def test_transactional_edits(workspace):
    editor = TransactionalFileEditor(workspace)
    content = "def test():\n    return True"
    staging_path = editor.stage_edit("test.py", content)
    assert editor.validate_staging(staging_path, "python")
    editor.commit_all()
    assert os.path.exists(os.path.join(workspace, "test.py"))

def test_poison_detection():
    detector = ContextPoisonDetector()
    poisoned_history = [
        {"failure_type": "ast_validation_failed"},
        {"failure_type": "ast_validation_failed"},
        {"failure_type": "ast_validation_failed"}
    ]
    assert detector.is_poisoned(poisoned_history, [])

def test_execution_journal():
    journal = ExecutionJournal("session_1")
    journal.record("reasoning", "openai", {"thought": "test"})
    assert len(journal.entries) == 1