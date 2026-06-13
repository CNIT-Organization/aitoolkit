"""Exception hierarchy for aitoolkit.

A single base class lets callers catch every toolkit-originated error with one
``except AIToolkitError``, while specific subclasses allow fine-grained handling.
"""

from __future__ import annotations


class AIToolkitError(Exception):
    """Base class for all aitoolkit errors."""


class ConfigurationError(AIToolkitError):
    """Raised when configuration is missing or invalid."""


class LLMError(AIToolkitError):
    """Raised when an LLM request fails or returns an unusable response."""


class EmbeddingsError(AIToolkitError):
    """Raised when an embeddings request fails."""


class STTError(AIToolkitError):
    """Raised when speech-to-text transcription fails."""


class TTSError(AIToolkitError):
    """Raised when text-to-speech synthesis fails."""


class VectorStoreError(AIToolkitError):
    """Raised when a vector-store operation fails."""
