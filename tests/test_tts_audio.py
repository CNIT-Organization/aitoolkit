"""Unit tests for WAV stitching (``aitoolkit.tts.audio.concat_wav``)."""

from __future__ import annotations

import io
import wave

import pytest

from aitoolkit.exceptions import TTSError
from aitoolkit.tts.audio import concat_wav


def _make_wav(*, seconds: float, framerate: int = 44100, nchannels: int = 1) -> bytes:
    """Build a silent WAV clip with the given format, for tests."""
    nframes = int(framerate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(2)  # 16-bit PCM
        writer.setframerate(framerate)
        writer.writeframes(b"\x00\x00" * nframes * nchannels)
    return buffer.getvalue()


def _duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def test_concat_sums_durations() -> None:
    out = concat_wav([_make_wav(seconds=1.0), _make_wav(seconds=0.5)])
    assert _duration(out) == pytest.approx(1.5, abs=0.01)


def test_concat_inserts_gap() -> None:
    out = concat_wav(
        [_make_wav(seconds=1.0), _make_wav(seconds=1.0)], gap_ms=500
    )
    # two 1.0s clips + one 0.5s gap between them
    assert _duration(out) == pytest.approx(2.5, abs=0.01)


def test_concat_skips_empty_segments() -> None:
    out = concat_wav([b"", _make_wav(seconds=1.0), b""])
    assert _duration(out) == pytest.approx(1.0, abs=0.01)


def test_concat_preserves_format() -> None:
    out = concat_wav([_make_wav(seconds=0.2, framerate=44100)])
    with wave.open(io.BytesIO(out), "rb") as reader:
        assert reader.getframerate() == 44100
        assert reader.getsampwidth() == 2
        assert reader.getnchannels() == 1


def test_concat_rejects_no_segments() -> None:
    with pytest.raises(TTSError):
        concat_wav([])
    with pytest.raises(TTSError):
        concat_wav([b"", b""])


def test_concat_rejects_mismatched_formats() -> None:
    with pytest.raises(TTSError, match="must share"):
        concat_wav([_make_wav(seconds=0.1, framerate=44100),
                    _make_wav(seconds=0.1, framerate=24000)])


def test_concat_rejects_invalid_wav() -> None:
    with pytest.raises(TTSError, match="not valid WAV"):
        concat_wav([b"not-a-wav-file"])
