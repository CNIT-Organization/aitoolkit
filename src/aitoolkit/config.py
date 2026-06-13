"""Central configuration for aitoolkit.

All settings are read from ``AITOOLKIT_*`` environment variables (or a ``.env``
file) but every client also accepts explicit overrides, so the package can be
used with zero environment configuration.

Defaults intentionally point at ``localhost`` — they are NOT specific to any one
deployment. Production endpoints are supplied by the consuming application via
environment variables (see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# A placeholder key for OpenAI-compatible servers that perform no app-layer auth
# (our GPU services are firewalled, not key-gated). The openai SDK requires a
# non-empty key, so we provide one.
_NO_AUTH = "no-auth"


class AIToolkitSettings(BaseSettings):
    """Runtime configuration for every aitoolkit capability."""

    model_config = SettingsConfigDict(
        env_prefix="AITOOLKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (vLLM, OpenAI-compatible) ---
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_api_key: str = Field(default=_NO_AUTH)
    # No default model id — set AITOOLKIT_LLM_MODEL to your served model.
    llm_model: str = Field(default="")
    llm_temperature: float = Field(default=0.2)
    llm_timeout: float = Field(default=60.0)
    llm_max_retries: int = Field(default=2)

    # --- Embeddings (TEI, OpenAI-compatible) ---
    embeddings_base_url: str = Field(default="http://localhost:8001/v1")
    embeddings_api_key: str = Field(default=_NO_AUTH)
    # No default model id — set AITOOLKIT_EMBEDDINGS_MODEL to your served model.
    embeddings_model: str = Field(default="")
    # TEI accepts modest batches; keep conservative and configurable.
    embeddings_batch_size: int = Field(default=32)
    embeddings_timeout: float = Field(default=60.0)

    # --- Speech-to-Text (faster-whisper, OpenAI-compatible) ---
    stt_base_url: str = Field(default="http://localhost:8003/v1")
    stt_api_key: str = Field(default=_NO_AUTH)
    stt_model: str = Field(default="whisper-1")
    stt_language: Optional[str] = Field(default=None)
    stt_timeout: float = Field(default=120.0)

    # --- Text-to-Speech (custom /api/tts) ---
    tts_base_url: str = Field(default="http://localhost:8002")
    tts_default_voice: Optional[str] = Field(default=None)
    tts_timeout: float = Field(default=120.0)

    # --- Vector store (Qdrant) ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="documents")
    # Optional fixed vector size. When None, it is detected from the embedding model.
    qdrant_vector_size: Optional[int] = Field(default=None)
    # The qdrant-client refuses to talk to a server whose minor version differs by
    # more than one, emitting a UserWarning. Self-hosted servers often lag the
    # client; set False to silence the check when the API surface we use is stable.
    qdrant_check_compatibility: bool = Field(default=True)

    # --- Retriever cache (Redis, optional) ---
    redis_url: Optional[str] = Field(default=None)
    cache_ttl: int = Field(default=3600)


@lru_cache(maxsize=1)
def get_settings() -> AIToolkitSettings:
    """Return the process-wide settings singleton."""
    return AIToolkitSettings()
