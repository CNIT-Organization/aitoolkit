"""Embeddings client backed by an OpenAI-compatible embeddings server (TEI).

Async-first, with synchronous convenience wrappers so it can also satisfy the
LangChain ``Embeddings`` interface. The embedding dimension is detected at
runtime — never hardcoded — so swapping the underlying model requires no code
change here (only a re-index in the vector store).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from aitoolkit.config import AIToolkitSettings, get_settings
from aitoolkit.exceptions import EmbeddingsError


class EmbeddingsClient:
    """Create embeddings via an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        settings: Optional[AIToolkitSettings] = None,
    ) -> None:
        settings = settings or get_settings()
        self.model = model or settings.embeddings_model
        self.batch_size = batch_size or settings.embeddings_batch_size
        self._base_url = base_url or settings.embeddings_base_url
        self._api_key = api_key or settings.embeddings_api_key
        self._timeout = timeout if timeout is not None else settings.embeddings_timeout

        self._aclient = AsyncOpenAI(
            base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
        )
        self._sclient: Optional[OpenAI] = None
        self._dimension: Optional[int] = None
        logger.info(
            f"EmbeddingsClient ready (model={self.model}, base_url={self._base_url})"
        )

    @property
    def sync_client(self) -> OpenAI:
        if self._sclient is None:
            self._sclient = OpenAI(
                base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
            )
        return self._sclient

    @property
    def dimension(self) -> Optional[int]:
        """The embedding dimension, known after the first call."""
        return self._dimension

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _batched(items: List[str], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _record_dim(self, vectors: List[List[float]]) -> None:
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
            logger.debug(f"Detected embedding dimension: {self._dimension}")

    # ----------------------------------------------------------------- async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed many documents, batching to respect server limits."""
        if not texts:
            return []
        out: List[List[float]] = []
        for batch in self._batched(texts, self.batch_size):
            try:
                resp = await self._aclient.embeddings.create(
                    model=self.model, input=batch
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingsError(f"embedding request failed: {exc}") from exc
            # Preserve input order via the returned index.
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in ordered])
        self._record_dim(out)
        return out

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        vectors = await self.aembed_documents([text])
        if not vectors:
            raise EmbeddingsError("empty embedding response for query")
        return vectors[0]

    # ------------------------------------------------------------------ sync
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous document embedding (LangChain ``Embeddings`` interface)."""
        if not texts:
            return []
        out: List[List[float]] = []
        for batch in self._batched(texts, self.batch_size):
            try:
                resp = self.sync_client.embeddings.create(
                    model=self.model, input=batch
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingsError(f"embedding request failed: {exc}") from exc
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in ordered])
        self._record_dim(out)
        return out

    def embed_query(self, text: str) -> List[float]:
        """Synchronous single-query embedding."""
        vectors = self.embed_documents([text])
        if not vectors:
            raise EmbeddingsError("empty embedding response for query")
        return vectors[0]

    async def aclose(self) -> None:
        await self._aclient.close()
        if self._sclient is not None:
            self._sclient.close()


@lru_cache(maxsize=1)
def get_embeddings_client() -> EmbeddingsClient:
    """Return the process-wide embeddings client singleton."""
    return EmbeddingsClient()
