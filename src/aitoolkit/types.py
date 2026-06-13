"""Plain, provider-agnostic data types used across aitoolkit.

These are deliberately simple so the public surface does not leak any third-party
SDK types to callers.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    """A single chat message in OpenAI message format."""

    role: Role
    content: str


class DialogueTurn(TypedDict):
    """One turn of a multi-speaker dialogue to synthesize: a voice and its line."""

    voice_id: str
    text: str


class TranscriptionResult(BaseModel):
    """Result of a speech-to-text transcription."""

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


class RetrievedChunk(BaseModel):
    """A single retrieved context chunk with score and metadata."""

    text: str
    score: float = 0.0
    file_id: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


def as_messages(
    prompt: Optional[str] = None,
    *,
    system: Optional[str] = None,
    messages: Optional[List[ChatMessage]] = None,
) -> List[ChatMessage]:
    """Normalize the various ways callers pass prompts into a message list.

    Accepts either an explicit ``messages`` list, or a ``prompt`` (plus optional
    ``system``) convenience form.
    """
    if messages is not None:
        return list(messages)

    out: List[ChatMessage] = []
    if system:
        out.append({"role": "system", "content": system})
    if prompt:
        out.append({"role": "user", "content": prompt})
    return out
