"""WAV audio helpers — stitch multiple WAV clips into one.

Pure standard library (``wave``); no third-party audio dependencies, keeping the
core package light. All input clips must share the same format (channels, sample
width, frame rate) — produce them with a single TTS engine / voice family.
"""

from __future__ import annotations

import io
import wave
from typing import List, Optional, Sequence

from aitoolkit.exceptions import TTSError


def concat_wav(segments: Sequence[bytes], *, gap_ms: int = 0) -> bytes:
    """Concatenate WAV byte clips into a single WAV.

    Args:
        segments: WAV-encoded audio clips. Empty clips are skipped. All
            non-empty clips must share channels, sample width and frame rate.
        gap_ms: silence inserted between consecutive clips, in milliseconds.

    Returns:
        A single WAV-encoded byte string.

    Raises:
        TTSError: if there are no usable segments or their formats differ.
    """
    clips = [clip for clip in segments if clip]
    if not clips:
        raise TTSError("concat_wav: no audio segments to concatenate")

    params: Optional[wave._wave_params] = None
    base_format: Optional[tuple] = None
    frames: List[bytes] = []

    for index, clip in enumerate(clips):
        try:
            with wave.open(io.BytesIO(clip), "rb") as reader:
                clip_params = reader.getparams()
                clip_frames = reader.readframes(clip_params.nframes)
        except (wave.Error, EOFError) as exc:
            raise TTSError(f"concat_wav: segment {index} is not valid WAV: {exc}") from exc

        clip_format = (clip_params.nchannels, clip_params.sampwidth, clip_params.framerate)
        if params is None:
            params, base_format = clip_params, clip_format
        elif clip_format != base_format:
            raise TTSError(
                f"concat_wav: segment {index} format {clip_format} != {base_format}; "
                "all clips must share channels/width/rate (synthesize with one engine)"
            )

        if frames and gap_ms > 0:
            silent_samples = int(params.framerate * gap_ms / 1000)
            frames.append(b"\x00" * silent_samples * params.nchannels * params.sampwidth)
        frames.append(clip_frames)

    assert params is not None  # guaranteed: clips is non-empty
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.writeframes(b"".join(frames))
    return buffer.getvalue()
