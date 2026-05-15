import pytest
from core.healing.taxonomy import FailureTaxonomy, FailureType, FailureSeverity

def test_failure_classification():
    # Transport failure
    fail = FailureTaxonomy.classify(TimeoutError("request timed out"))
    assert fail.type == FailureType.PROVIDER_TIMEOUT
    assert fail.severity == FailureSeverity.MEDIUM
    
    # AST failure
    fail = FailureTaxonomy.classify(SyntaxError("invalid syntax"))
    assert fail.type == FailureType.AST_VALIDATION_FAILED
    assert fail.severity == FailureSeverity.MEDIUM
    
    # Unknown
    fail = FailureTaxonomy.classify(ValueError("something weird"))
    assert fail.type == FailureType.UNKNOWN
    assert fail.severity == FailureSeverity.HIGH
