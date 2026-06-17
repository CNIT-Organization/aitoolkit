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
from aitoolkit.retry import retry_async
from aitoolkit.tts.audio import concat_wav
from aitoolkit.types import DialogueTurn

# HTTP statuses worth retrying (transient). Other 4xx are caller errors.
_RETRIABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class _RetriableTTS(Exception):
    """Internal marker for a transient TTS failure (timeout / 5xx / 429)."""


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

        # Fast connect (fail quickly + retry), longer read for the actual synthesis.
        timeout = httpx.Timeout(self._timeout, connect=5.0)

        async def _attempt() -> bytes:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
            except httpx.TransportError as exc:
                # Transient transport faults — covers all timeouts (TimeoutException)
                # and network/connection errors (ConnectError, ReadError, …).
                raise _RetriableTTS(f"{type(exc).__name__}: {exc}") from exc
            except httpx.HTTPError as exc:  # other httpx errors — don't retry.
                raise TTSError(
                    f"TTS request failed: {type(exc).__name__}: {exc}"
                ) from exc

            if resp.status_code != 200:
                detail = self._error_detail(resp)
                if resp.status_code in _RETRIABLE_STATUS:
                    raise _RetriableTTS(detail)
                raise TTSError(detail)  # 4xx caller error — don't retry.

            if not resp.content:
                raise TTSError("TTS returned an empty audio response")
            return resp.content

        try:
            return await retry_async(
                _attempt, retry_on=(_RetriableTTS,), label="TTS synthesize"
            )
        except _RetriableTTS as exc:
            # Exhausted retries on a transient fault — surface the real reason.
            raise TTSError(f"TTS request failed after retries: {exc}") from exc

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

        # Each turn already retries transient faults (see ``synthesize``). If a
        # turn STILL fails, skip it rather than fail the whole dialogue — a
        # podcast missing one line is far better than no podcast at all. We only
        # fail if every turn failed.
        clips: List[bytes] = []
        failed = 0
        for index, turn in enumerate(spoken):
            try:
                clips.append(
                    await self.synthesize(
                        turn["text"],
                        voice=turn["voice_id"],
                        language=language,
                        **kwargs,
                    )
                )
            except TTSError as exc:
                failed += 1
                logger.warning(
                    f"synthesize_dialogue: skipping turn {index + 1}/{len(spoken)} "
                    f"after retries failed ({exc})"
                )

        if not clips:
            raise TTSError(
                f"synthesize_dialogue: all {len(spoken)} turns failed to synthesize"
            )
        if failed:
            logger.info(
                f"synthesize_dialogue: produced {len(clips)}/{len(spoken)} turns "
                f"({failed} skipped after retries)"
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
