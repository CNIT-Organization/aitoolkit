"""LangChain bridge (optional extra ``aitoolkit[langchain]``).

Provides:

* :func:`to_chat_model` — a LangChain ``BaseChatModel`` pointed at the toolkit's
  LLM endpoint, so existing LangGraph graphs keep working unchanged.
* :class:`LangChainEmbeddings` — wraps :class:`EmbeddingsClient` as a LangChain
  ``Embeddings`` so vector stores / chains can consume toolkit embeddings.

Only this module imports LangChain; the toolkit core stays LangChain-free.
"""

from __future__ import annotations

from typing import List, Optional

try:  # pragma: no cover - import guard
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain integration requires extra deps. "
        "Install with: pip install 'aitoolkit[langchain]'"
    ) from exc

from aitoolkit.config import get_settings
from aitoolkit.embeddings import EmbeddingsClient, get_embeddings_client


def to_chat_model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
    **kwargs,
) -> ChatOpenAI:
    """Return a LangChain ``ChatOpenAI`` bound to the toolkit's LLM endpoint."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        streaming=streaming,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        **kwargs,
    )


class LangChainEmbeddings(Embeddings):
    """Adapt :class:`EmbeddingsClient` to the LangChain ``Embeddings`` interface."""

    def __init__(self, client: Optional[EmbeddingsClient] = None) -> None:
        self._client = client or get_embeddings_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._client.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self._client.aembed_query(text)


__all__ = ["to_chat_model", "LangChainEmbeddings"]
