"""Resilience tests for the TTS client: transient retries, no-retry on 4xx, and
graceful per-turn skipping in ``synthesize_dialogue``."""

from __future__ import annotations

import io
import wave

import httpx
import pytest
import respx

from aitoolkit.config import AIToolkitSettings
from aitoolkit.exceptions import TTSError
from aitoolkit.tts.client import TTSClient

BASE = "http://tts.test"
TTS_URL = f"{BASE}/api/tts"


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip retry backoff sleeps so tests run instantly."""

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("aitoolkit.retry.asyncio.sleep", _noop)


def _client() -> TTSClient:
    # Pass an explicit settings object so the test never reads env/config.
    return TTSClient(
        base_url=BASE, default_voice="M1", timeout=1.0, settings=AIToolkitSettings()
    )


def _wav(seconds: float = 0.1) -> bytes:
    """A tiny valid WAV so concat_wav can parse synthesized turns."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


@respx.mock
async def test_synthesize_retries_5xx_then_succeeds(no_sleep) -> None:
    route = respx.post(TTS_URL).mock(
        side_effect=[
            httpx.Response(503, json={"detail": "busy"}),
            httpx.Response(200, content=b"AUDIO"),
        ]
    )
    audio = await _client().synthesize("hello", voice="M1")
    assert audio == b"AUDIO"
    assert route.call_count == 2  # retried the 503, then succeeded


@respx.mock
async def test_synthesize_retries_read_timeout_then_succeeds(no_sleep) -> None:
    route = respx.post(TTS_URL).mock(
        side_effect=[
            httpx.ReadTimeout("server too slow"),
            httpx.Response(200, content=b"AUDIO"),
        ]
    )
    audio = await _client().synthesize("hello", voice="M1")
    assert audio == b"AUDIO"
    assert route.call_count == 2


@respx.mock
async def test_synthesize_does_not_retry_4xx(no_sleep) -> None:
    route = respx.post(TTS_URL).mock(return_value=httpx.Response(400, json={"detail": "bad"}))
    with pytest.raises(TTSError):
        await _client().synthesize("hello", voice="M1")
    assert route.call_count == 1  # 4xx is a caller error -> not retried


@respx.mock
async def test_synthesize_gives_up_after_attempts(no_sleep) -> None:
    route = respx.post(TTS_URL).mock(return_value=httpx.Response(503, json={"detail": "busy"}))
    with pytest.raises(TTSError, match="after retries"):
        await _client().synthesize("hello", voice="M1")
    assert route.call_count == 3  # default attempts


@respx.mock
async def test_dialogue_skips_failed_turn(no_sleep) -> None:
    # turn 1 ok, turn 2 fails (4xx, not retried -> skipped), turn 3 ok.
    route = respx.post(TTS_URL).mock(
        side_effect=[
            httpx.Response(200, content=_wav()),
            httpx.Response(400, json={"detail": "bad turn"}),
            httpx.Response(200, content=_wav()),
        ]
    )
    turns = [
        {"voice_id": "M1", "text": "one"},
        {"voice_id": "F1", "text": "two"},
        {"voice_id": "M1", "text": "three"},
    ]
    out = await _client().synthesize_dialogue(turns, gap_ms=0)
    assert out  # a valid (non-empty) WAV from the 2 surviving turns
    assert route.call_count == 3  # all three attempted; the middle one skipped


@respx.mock
async def test_dialogue_all_turns_fail_raises(no_sleep) -> None:
    respx.post(TTS_URL).mock(return_value=httpx.Response(400, json={"detail": "bad"}))
    turns = [
        {"voice_id": "M1", "text": "one"},
        {"voice_id": "F1", "text": "two"},
    ]
    with pytest.raises(TTSError, match="all .* turns failed"):
        await _client().synthesize_dialogue(turns)
