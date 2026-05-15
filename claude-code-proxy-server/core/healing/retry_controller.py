"""Adaptive retry controller with provider-specific cooldowns and jitter."""

from __future__ import annotations
import asyncio
import random
import time
from typing import Dict, Optional
from loguru import logger
from .taxonomy import FailureType, FailureClassification

class AdaptiveRetryController:
    """Manages dynamic backoff and provider cooldowns."""

    def __init__(self):
        self._provider_last_failure: Dict[str, float] = {}
        self._provider_failure_count: Dict[str, int] = {}
        self._base_delay = 1.0
        self._max_delay = 60.0

    async def wait_before_retry(self, provider: str, failure: FailureClassification) -> None:
        """Calculate and wait for the appropriate backoff period."""
        
        # 1. Update failure tracking
        now = time.time()
        self._provider_last_failure[provider] = now
        self._provider_failure_count[provider] = self._provider_failure_count.get(provider, 0) + 1
        
        count = self._provider_failure_count[provider]
        
        # 2. Calculate exponential backoff
        delay = min(self._max_delay, self._base_delay * (2 ** (count - 1)))
        
        # 3. Apply failure-type specific logic
        if failure.type == FailureType.RATE_LIMIT_EXCEEDED:
            # Stricter backoff for rate limits
            delay = max(delay, 5.0)
        elif failure.type == FailureType.PROVIDER_TIMEOUT:
            # Shorter backoff for timeouts as it might be a transient glitch
            delay = min(delay, 10.0)

        # 4. Add dynamic jitter (0.8x to 1.2x) to prevent thundering herd
        jitter = random.uniform(0.8, 1.2)
        final_delay = delay * jitter
        
        logger.info(
            "RETRY_CONTROLLER: Backing off for {:.2f}s on provider {} (failure_count={}, type={})",
            final_delay, provider, count, failure.type
        )
        
        await asyncio.sleep(final_delay)

    def reset_provider(self, provider: str) -> None:
        """Reset failure tracking for a provider after success."""
        if provider in self._provider_failure_count:
            logger.debug("RETRY_CONTROLLER: Resetting backoff for provider {}", provider)
            self._provider_failure_count[provider] = 0

    def is_provider_cooling_down(self, provider: str) -> bool:
        """Check if a provider is in a lock-out period after critical failures."""
        last_fail = self._provider_last_failure.get(provider, 0)
        count = self._provider_failure_count.get(provider, 0)
        
        if count >= 5: # Critical failure threshold
            cooldown = 30.0 # 30s lockout
            if time.time() - last_fail < cooldown:
                return True
        return False
