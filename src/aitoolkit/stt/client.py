"""Speech-to-text client backed by an OpenAI-compatible transcription endpoint.

Targets the self-hosted faster-whisper service which exposes
``/v1/audio/transcriptions``. Replaces any local/in-process Whisper so no model
weights are loaded inside the application.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Optional, Union

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from aitoolkit.config import AIToolkitSettings, get_settings
from aitoolkit.exceptions import STTError
from aitoolkit.types import TranscriptionResult

AudioInput = Union[str, Path, bytes, BinaryIO]


class STTClient:
    """Transcribe audio via an OpenAI-compatible ``audio.transcriptions`` API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        settings: Optional[AIToolkitSettings] = None,
    ) -> None:
        settings = settings or get_settings()
        self.model = model or settings.stt_model
        self.default_language = language or settings.stt_language
        self._base_url = base_url or settings.stt_base_url
        self._api_key = api_key or settings.stt_api_key
        self._timeout = timeout if timeout is not None else settings.stt_timeout

        self._aclient = AsyncOpenAI(
            base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
        )
        self._sclient: Optional[OpenAI] = None
        logger.info(
            f"STTClient ready (model={self.model}, base_url={self._base_url})"
        )

    @property
    def sync_client(self) -> OpenAI:
        if self._sclient is None:
            self._sclient = OpenAI(
                base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
            )
        return self._sclient

    @staticmethod
    def _to_file(audio: AudioInput):
        """Normalize various audio inputs into something the SDK accepts."""
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            return (path.name, path.read_bytes())
        if isinstance(audio, bytes):
            return ("audio.wav", audio)
        if isinstance(audio, io.IOBase) or hasattr(audio, "read"):
            data = audio.read()
            name = getattr(audio, "name", "audio.wav")
            return (Path(str(name)).name, data)
        raise STTError(f"unsupported audio input type: {type(audio)!r}")

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio and return text plus optional metadata.

        ``response_format`` defaults to ``"json"`` (text only). Pass
        ``"verbose_json"`` to also populate ``language`` and ``duration`` on the
        returned :class:`TranscriptionResult`.
        """
        file_arg = self._to_file(audio)
        try:
            resp = await self._aclient.audio.transcriptions.create(
                file=file_arg,
                model=self.model,
                language=language or self.default_language,
                prompt=prompt,
                response_format=response_format,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise STTError(f"transcription failed: {exc}") from exc
        return self._to_result(resp, language or self.default_language)

    def transcribe_sync(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        **kwargs,
    ) -> TranscriptionResult:
        """Synchronous counterpart of :meth:`transcribe`."""
        file_arg = self._to_file(audio)
        try:
            resp = self.sync_client.audio.transcriptions.create(
                file=file_arg,
                model=self.model,
                language=language or self.default_language,
                prompt=prompt,
                response_format=response_format,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise STTError(f"transcription failed: {exc}") from exc
        return self._to_result(resp, language or self.default_language)

    @staticmethod
    def _to_result(resp, language: Optional[str]) -> TranscriptionResult:
        text = getattr(resp, "text", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("text", "")
        return TranscriptionResult(
            text=text or "",
            language=getattr(resp, "language", None) or language,
            duration=getattr(resp, "duration", None),
        )

    async def aclose(self) -> None:
        await self._aclient.close()
        if self._sclient is not None:
            self._sclient.close()


@lru_cache(maxsize=1)
def get_stt_client() -> STTClient:
    """Return the process-wide STT client singleton."""
    return STTClient()
