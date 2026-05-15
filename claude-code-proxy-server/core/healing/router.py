"""Capability-aware routing for selecting recovery providers."""

from __future__ import annotations
from typing import List, Optional, Dict
from loguru import logger
from .normalization import ProviderProfile
from .taxonomy import FailureType, FailureClassification

class CapabilityAwareRouter:
    """Routes recovery requests to the most capable provider for the failure type."""

    def __init__(self, profiles: List[ProviderProfile]):
        self.profiles = {p.name: p for p in profiles}

    def select_recovery_provider(
        self, 
        current_provider: str, 
        failure: FailureClassification
    ) -> ProviderProfile:
        """Select the best provider to handle the recovery of the given failure."""
        
        # 1. If it's a transient transport failure, we prefer staying on the current provider
        if failure.type in (FailureType.CONNECTION_INTERRUPTED, FailureType.PROVIDER_TIMEOUT):
            if current_provider in self.profiles:
                return self.profiles[current_provider]

        # 2. If it's a semantic/code failure, we want the most "intelligent" model
        if failure.type == FailureType.AST_VALIDATION_FAILED:
            # Sort by ast_correction_score
            sorted_profiles = sorted(
                self.profiles.values(), 
                key=lambda p: p.ast_correction_score, 
                reverse=True
            )
            # Pick the top one that isn't the failing one (if possible)
            for p in sorted_profiles:
                if p.name != current_provider:
                    logger.info("ROUTER: Routing code repair to high-fidelity model: {}", p.name)
                    return p
            return sorted_profiles[0]

        # 3. Fallback: return current or first available
        return self.profiles.get(current_provider) or next(iter(self.profiles.values()))

    def get_failover_candidates(self, failing_provider: str) -> List[ProviderProfile]:
        """Get a list of potential failover providers, excluded the failing one."""
        return [p for p in self.profiles.values() if p.name != failing_provider]
