"""Text-to-speech client for the custom self-hosted TTS services.

Unlike LLM/embeddings/STT, the TTS servers are **not** OpenAI-compatible: they
expose ``POST /api/tts`` (returning raw audio bytes) and ``GET /api/voices``.
A successful synthesis returns binary audio; errors return JSON ``{"detail": ...}``.
The request requires either a ``voice_id`` or an ``instruct`` prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Union

import httpx
from loguru import logger

from aitoolkit.config import AIToolkitSettings, get_settings
from aitoolkit.exceptions import TTSError
from aitoolkit.tts.audio import concat_wav
from aitoolkit.types import DialogueTurn


class TTSClient:
    """Synthesize speech via a custom ``/api/tts`` endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_voice: Optional[str] = None,
        timeout: Optional[float] = None,
        tts_path: str = "/api/tts",
        voices_path: str = "/api/voices",
        settings: Optional[AIToolkitSettings] = None,
    ) -> None:
        settings = settings or get_settings()
        self._base_url = (base_url or settings.tts_base_url).rstrip("/")
        self.default_voice = default_voice or settings.tts_default_voice
        self._timeout = timeout if timeout is not None else settings.tts_timeout
        self._tts_path = tts_path
        self._voices_path = voices_path
        logger.info(f"TTSClient ready (base_url={self._base_url})")

    async def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: str = "en",
        instruct: Optional[str] = None,
        speed: Optional[float] = None,
        num_step: Optional[int] = None,
        ref_text: Optional[str] = None,
    ) -> bytes:
        """Synthesize ``text`` and return raw audio bytes.

        Either ``voice`` (resolved against ``default_voice``) or ``instruct``
        must be provided, matching the server contract.
        """
        voice_id = voice or self.default_voice
        if not voice_id and not instruct:
            raise TTSError("either 'voice' (voice_id) or 'instruct' must be provided")

        payload: dict = {"text": text, "language": language}
        if voice_id:
            payload["voice_id"] = voice_id
        if instruct is not None:
            payload["instruct"] = instruct
        if speed is not None:
            payload["speed"] = speed
        if num_step is not None:
            payload["num_step"] = num_step
        if ref_text is not None:
            payload["ref_text"] = ref_text

        url = f"{self._base_url}{self._tts_path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise TTSError(f"TTS request failed: {exc}") from exc

        if resp.status_code != 200:
            raise TTSError(self._error_detail(resp))

        audio = resp.content
        if not audio:
            raise TTSError("TTS returned an empty audio response")
        return audio

    async def synthesize_to_file(
        self, text: str, path: Union[str, Path], **kwargs
    ) -> Path:
        """Synthesize and write the audio to ``path``; returns the path."""
        audio = await self.synthesize(text, **kwargs)
        out = Path(path)
        out.write_bytes(audio)
        return out

    async def synthesize_dialogue(
        self,
        turns: Sequence[DialogueTurn],
        *,
        language: str = "en",
        gap_ms: int = 300,
        **kwargs,
    ) -> bytes:
        """Synthesize a multi-speaker dialogue into a single WAV.

        Each turn is synthesized with its own ``voice_id`` and the clips are
        concatenated with a short silent gap between turns. All voices must live
        on the same TTS engine so the clips share an audio format (see
        :func:`aitoolkit.tts.audio.concat_wav`).

        Args:
            turns: ordered ``{"voice_id", "text"}`` turns. Empty texts are skipped.
            language: language code passed to each synthesis.
            gap_ms: silence inserted between turns, in milliseconds.
            **kwargs: forwarded to :meth:`synthesize` (e.g. ``speed``).

        Returns:
            A single WAV-encoded byte string.

        Raises:
            TTSError: if no turns have text, or synthesis/concatenation fails.
        """
        spoken = [turn for turn in turns if turn.get("text", "").strip()]
        if not spoken:
            raise TTSError("synthesize_dialogue: no non-empty turns provided")

        clips: List[bytes] = []
        for turn in spoken:
            clips.append(
                await self.synthesize(
                    turn["text"],
                    voice=turn["voice_id"],
                    language=language,
                    **kwargs,
                )
            )
        return concat_wav(clips, gap_ms=gap_ms)

    async def list_voices(self) -> List[dict]:
        """Return the available voices from the server."""
        url = f"{self._base_url}{self._voices_path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise TTSError(f"failed to list voices: {exc}") from exc
        return data.get("voices", data) if isinstance(data, dict) else data

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except Exception:  # noqa: BLE001 - body may be non-JSON
            detail = resp.text[:200]
        return f"TTS failed (HTTP {resp.status_code}): {detail}"


@lru_cache(maxsize=1)
def get_tts_client() -> TTSClient:
    """Return the process-wide TTS client singleton."""
    return TTSClient()
