"""Failure taxonomy for classifying and routing healing strategies."""

from __future__ import annotations
from enum import StrEnum
from typing import Dict, Any, Optional
from pydantic import BaseModel

class FailureType(StrEnum):
    """Categories of failures that require different healing strategies."""
    
    # Transport Level
    CONNECTION_INTERRUPTED = "connection_interrupted"
    PROVIDER_TIMEOUT = "provider_timeout"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    PROVIDER_5XX = "provider_5xx"
    
    # Content/Integrity Level
    MALFORMED_SSE = "malformed_sse"
    STREAM_INTEGRITY_VIOLATION = "stream_integrity_violation"
    TRUNCATED_TOOL_CALL = "truncated_tool_call"
    
    # Semantic/Cognitive Level
    AST_VALIDATION_FAILED = "ast_validation_failed"
    LINT_CHECK_FAILED = "lint_check_failed"
    BUILD_FAILED = "build_failed"
    RECURSIVE_REPAIR_DETECTED = "recursive_repair_detected" # Context poisoning
    SEMANTIC_DRIFT_DETECTED = "semantic_drift_detected"
    
    # Generic
    UNKNOWN = "unknown"

class FailureSeverity(StrEnum):
    LOW = "low"      # Retry immediately
    MEDIUM = "medium" # Requires semantic resumption
    HIGH = "high"    # Requires rollback or provider switch
    CRITICAL = "critical" # Requires human intervention or full reset

class FailureClassification(BaseModel):
    """Detailed classification of a failure instance."""
    type: FailureType
    severity: FailureSeverity
    message: str
    context: Dict[str, Any] = {}
    can_retry: bool = True
    suggested_strategy: Optional[str] = None

class FailureTaxonomy:
    """Logic for classifying raw errors into the taxonomy."""
    
    @staticmethod
    def classify(error: Exception, context: Optional[Dict[str, Any]] = None) -> FailureClassification:
        error_name = type(error).__name__
        error_msg = str(error).lower()
        ctx = context or {}

        # 1. Transport Failures
        if "timeout" in error_msg or "timed out" in error_msg or "deadline" in error_msg:
            return FailureClassification(
                type=FailureType.PROVIDER_TIMEOUT,
                severity=FailureSeverity.MEDIUM,
                message="Request timed out",
                can_retry=True
            )
        
        if "connection" in error_msg or "broken pipe" in error_msg or isinstance(error, ConnectionError):
            return FailureClassification(
                type=FailureType.CONNECTION_INTERRUPTED,
                severity=FailureSeverity.MEDIUM,
                message="Connection lost mid-stream",
                can_retry=True
            )

        if "429" in error_msg or "rate limit" in error_msg:
            return FailureClassification(
                type=FailureType.RATE_LIMIT_EXCEEDED,
                severity=FailureSeverity.MEDIUM,
                message="Provider rate limit exceeded",
                can_retry=True
            )

        # 2. Integrity Failures (often passed via context from StreamIntegrityValidator)
        if ctx.get("is_integrity_failure"):
            return FailureClassification(
                type=FailureType.STREAM_INTEGRITY_VIOLATION,
                severity=FailureSeverity.MEDIUM,
                message=error_msg,
                can_retry=True,
                suggested_strategy="semantic_resumption"
            )

        # 3. Code/Semantic Failures
        if "syntax" in error_msg or "parse" in error_msg:
            return FailureClassification(
                type=FailureType.AST_VALIDATION_FAILED,
                severity=FailureSeverity.MEDIUM,
                message="Generated code has syntax errors",
                can_retry=True,
                suggested_strategy="repair_loop"
            )

        return FailureClassification(
            type=FailureType.UNKNOWN,
            severity=FailureSeverity.HIGH,
            message=f"Unclassified error: {error_name}: {error_msg}",
            can_retry=False
        )