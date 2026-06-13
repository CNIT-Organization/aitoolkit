"""Text-to-speech capability — custom self-hosted ``/api/tts`` server.

Works with any TTS service exposing the ``/api/tts`` + ``/api/voices`` contract
(``POST /api/tts`` returns raw audio bytes; ``GET /api/voices`` lists voices).
"""

from aitoolkit.tts.audio import concat_wav
from aitoolkit.tts.client import TTSClient, get_tts_client

__all__ = ["TTSClient", "get_tts_client", "concat_wav"]
