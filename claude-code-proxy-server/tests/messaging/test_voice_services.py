from pathlib import Path
from unittest.mock import patch

import pytest

from messaging.voice import PendingVoiceRegistry, VoiceTranscriptionService


@pytest.mark.asyncio
async def test_pending_voice_registry_tracks_voice_and_status_ids():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")

    assert await registry.is_pending("chat", "voice-1") is True
    assert await registry.cancel("chat", "status-1") == ("voice-1", "status-1")
    assert await registry.is_pending("chat", "voice-1") is False


@pytest.mark.asyncio
async def test_pending_voice_registry_complete_removes_entries():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")
    await registry.complete("chat", "voice-1", "status-1")

    assert await registry.cancel("chat", "voice-1") is None


@pytest.mark.asyncio
async def test_voice_transcription_service_runs_backend():
    service = VoiceTranscriptionService()

    with patch("messaging.transcription.transcribe_audio", return_value="hello"):
        text = await service.transcribe(
            Path("audio.ogg"),
            "audio/ogg",
            whisper_model="base",
            whisper_device="cpu",
        )

    assert text == "hello"


class _FakeLimiterSlot:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self):
        self._events.append("slot_enter")

    async def __aexit__(self, exc_type, exc, tb):
        self._events.append("slot_exit")
        return False


class _FakeLimiter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def concurrency_slot(self) -> _FakeLimiterSlot:
        return _FakeLimiterSlot(self.events)

    async def wait_if_blocked(self) -> bool:
        self.events.append("wait")
        return False


@pytest.mark.asyncio
async def test_voice_transcription_service_limits_nvidia_nim_before_backend():
    limiter = _FakeLimiter()

    with patch(
        "messaging.voice.GlobalRateLimiter.get_scoped_instance",
        return_value=limiter,
    ) as get_limiter:
        service = VoiceTranscriptionService(
            nvidia_nim_api_key="key",
            provider_rate_limit=40,
            provider_rate_window=60,
            provider_max_concurrency=5,
            nvidia_nim_rate_limit_headroom=2,
        )

        def transcribe(*args, **kwargs):
            limiter.events.append("backend")
            return "nim transcript"

        with patch("messaging.transcription.transcribe_audio", side_effect=transcribe):
            text = await service.transcribe(
                Path("audio.ogg"),
                "audio/ogg",
                whisper_model="openai/whisper-large-v3",
                whisper_device="nvidia_nim",
            )

    get_limiter.assert_called_once_with(
        "nim",
        rate_limit=38,
        rate_window=60,
        max_concurrency=5,
    )
    assert text == "nim transcript"
    assert limiter.events == ["slot_enter", "wait", "backend", "slot_exit"]


@pytest.mark.asyncio
async def test_voice_transcription_service_does_not_wait_for_local_whisper():
    service = VoiceTranscriptionService()

    with (
        patch("messaging.voice.GlobalRateLimiter.get_scoped_instance") as get_limiter,
        patch("messaging.transcription.transcribe_audio", return_value="local"),
    ):
        text = await service.transcribe(
            Path("audio.ogg"),
            "audio/ogg",
            whisper_model="base",
            whisper_device="cpu",
        )

    assert text == "local"
    get_limiter.assert_not_called()
