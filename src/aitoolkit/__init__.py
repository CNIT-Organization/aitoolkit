"""aitoolkit — centralized AI clients for self-hosted OpenAI-compatible services.

Core capabilities (LLM, embeddings, STT, TTS) are always available. RAG and the
LangChain bridge live behind extras and are imported from their own subpackages
(``aitoolkit.rag``, ``aitoolkit.integrations.langchain``) so that importing the
top-level package never forces an optional dependency.
"""

from __future__ import annotations

from aitoolkit.config import AIToolkitSettings, configure, get_settings
from aitoolkit.embeddings import EmbeddingsClient, get_embeddings_client
from aitoolkit.exceptions import (
    AIToolkitError,
    ConfigurationError,
    EmbeddingsError,
    LLMError,
    STTError,
    TTSError,
    VectorStoreError,
)
from aitoolkit.llm import LLMClient, get_llm_client
from aitoolkit.stt import STTClient, get_stt_client
from aitoolkit.tts import TTSClient, concat_wav, get_tts_client
from aitoolkit.types import (
    ChatMessage,
    DialogueTurn,
    RetrievedChunk,
    TranscriptionResult,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # config
    "AIToolkitSettings",
    "configure",
    "get_settings",
    # llm
    "LLMClient",
    "get_llm_client",
    # embeddings
    "EmbeddingsClient",
    "get_embeddings_client",
    # stt
    "STTClient",
    "get_stt_client",
    # tts
    "TTSClient",
    "get_tts_client",
    "concat_wav",
    # types
    "ChatMessage",
    "DialogueTurn",
    "RetrievedChunk",
    "TranscriptionResult",
    # exceptions
    "AIToolkitError",
    "ConfigurationError",
    "LLMError",
    "EmbeddingsError",
    "STTError",
    "TTSError",
    "VectorStoreError",
]
