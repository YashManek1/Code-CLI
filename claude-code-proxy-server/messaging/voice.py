"""Platform-neutral voice note helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from providers.rate_limit import GlobalRateLimiter

_NVIDIA_NIM_LIMITER_SCOPE = "nim"


def _effective_nim_rate_limit(provider_rate_limit: int, headroom: int) -> int:
    """Keep NIM traffic just below the configured provider quota."""
    return max(1, int(provider_rate_limit) - max(0, int(headroom)))


class PendingVoiceRegistry:
    """Track voice notes that are still waiting on transcription."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            entry = (voice_msg_id, status_msg_id)
            self._pending[(chat_id, voice_msg_id)] = entry
            self._pending[(chat_id, status_msg_id)] = entry

    async def cancel(self, chat_id: str, reply_id: str) -> tuple[str, str] | None:
        async with self._lock:
            entry = self._pending.pop((chat_id, reply_id), None)
            if entry is None:
                return None
            voice_msg_id, status_msg_id = entry
            self._pending.pop((chat_id, voice_msg_id), None)
            self._pending.pop((chat_id, status_msg_id), None)
            return entry

    async def is_pending(self, chat_id: str, voice_msg_id: str) -> bool:
        async with self._lock:
            return (chat_id, voice_msg_id) in self._pending

    async def complete(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            self._pending.pop((chat_id, voice_msg_id), None)
            self._pending.pop((chat_id, status_msg_id), None)


class VoiceTranscriptionService:
    """Run configured transcription backends off the event loop."""

    def __init__(
        self,
        *,
        hf_token: str = "",
        nvidia_nim_api_key: str = "",
        provider_rate_limit: int = 40,
        provider_rate_window: int = 60,
        provider_max_concurrency: int = 5,
        nvidia_nim_rate_limit_headroom: int = 10,
    ) -> None:
        self._hf_token = hf_token
        self._nvidia_nim_api_key = nvidia_nim_api_key
        self._provider_rate_limit = provider_rate_limit
        self._provider_rate_window = provider_rate_window
        self._provider_max_concurrency = provider_max_concurrency
        self._nvidia_nim_rate_limit_headroom = nvidia_nim_rate_limit_headroom
        self._nim_limiter: GlobalRateLimiter | None = None

    def _get_nim_limiter(self) -> GlobalRateLimiter:
        if self._nim_limiter is None:
            self._nim_limiter = GlobalRateLimiter.get_scoped_instance(
                _NVIDIA_NIM_LIMITER_SCOPE,
                rate_limit=_effective_nim_rate_limit(
                    self._provider_rate_limit,
                    self._nvidia_nim_rate_limit_headroom,
                ),
                rate_window=self._provider_rate_window,
                max_concurrency=self._provider_max_concurrency,
            )
        return self._nim_limiter

    async def transcribe(
        self,
        file_path: Path,
        mime_type: str,
        *,
        whisper_model: str,
        whisper_device: str,
    ) -> str:
        from .transcription import transcribe_audio

        if whisper_device == "nvidia_nim":
            nim_limiter = self._get_nim_limiter()
            async with nim_limiter.concurrency_slot():
                await nim_limiter.wait_if_blocked()
                return await asyncio.to_thread(
                    transcribe_audio,
                    file_path,
                    mime_type,
                    whisper_model=whisper_model,
                    whisper_device=whisper_device,
                    hf_token=self._hf_token,
                    nvidia_nim_api_key=self._nvidia_nim_api_key,
                )

        return await asyncio.to_thread(
            transcribe_audio,
            file_path,
            mime_type,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            hf_token=self._hf_token,
            nvidia_nim_api_key=self._nvidia_nim_api_key,
        )
