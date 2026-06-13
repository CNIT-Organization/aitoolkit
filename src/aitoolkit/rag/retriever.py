"""RAG retriever with optional Redis caching and pluggable filtering.

Differences from the old implementation:

* No LangChain dependency — operates on :class:`RetrievedChunk`.
* The hardcoded "drop DOCX files" rule is gone; pass a ``filter_fn`` predicate to
  apply any project-specific filtering instead.
* Redis is optional (``aitoolkit[cache]``); without it, caching is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, List, Optional

from loguru import logger

from aitoolkit.rag.vector_store import UnifiedVectorStore
from aitoolkit.types import RetrievedChunk

try:  # optional dependency
    import redis.asyncio as redis
except ImportError:  # pragma: no cover
    redis = None  # type: ignore[assignment]

FilterFn = Callable[[RetrievedChunk], bool]


class RAGRetriever:
    """Retrieve relevant chunks, with optional caching, filtering and reranking."""

    def __init__(
        self,
        vector_store: UnifiedVectorStore,
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600,
        filter_fn: Optional[FilterFn] = None,
        reranker: Optional[Callable[[str, List[RetrievedChunk]], List[RetrievedChunk]]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.cache_ttl = cache_ttl
        self.filter_fn = filter_fn
        self.reranker = reranker

        self.redis_client = None
        if redis_url and redis is not None:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("RAG retriever caching enabled")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"failed to init Redis cache: {exc}")
        elif redis_url and redis is None:
            logger.warning(
                "redis_url provided but redis not installed; install aitoolkit[cache]"
            )

    def _cache_key(self, query: str, file_ids: Optional[List[str]], limit: int) -> str:
        payload = {
            "q": query,
            "f": sorted(file_ids) if file_ids else None,
            "l": limit,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"aitoolkit:rag:{digest}"

    async def _cache_get(self, key: str) -> Optional[List[RetrievedChunk]]:
        if not self.redis_client:
            return None
        try:
            raw = await self.redis_client.get(key)
            if raw:
                return [RetrievedChunk(**item) for item in json.loads(raw)]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"cache read failed: {exc}")
        return None

    async def _cache_set(self, key: str, chunks: List[RetrievedChunk]) -> None:
        if not self.redis_client:
            return
        try:
            raw = json.dumps([c.model_dump() for c in chunks])
            await self.redis_client.setex(key, self.cache_ttl, raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"cache write failed: {exc}")

    async def retrieve(
        self,
        query: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[RetrievedChunk]:
        """Return relevant chunks for ``query``."""
        if use_cache:
            cached = await self._cache_get(self._cache_key(query, file_ids, limit))
            if cached is not None:
                logger.debug(f"cache hit ({len(cached)} chunks)")
                return cached

        chunks = await self.vector_store.similarity_search(
            query, file_ids=file_ids, limit=limit, score_threshold=score_threshold
        )

        if self.filter_fn:
            chunks = [c for c in chunks if self.filter_fn(c)]
        if self.reranker and chunks:
            chunks = self.reranker(query, chunks)

        if use_cache and chunks:
            await self._cache_set(self._cache_key(query, file_ids, limit), chunks)
        return chunks

    async def get_context_text(
        self,
        query: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        separator: str = "\n\n---\n\n",
        include_sources: bool = True,
    ) -> str:
        """Retrieve and format chunks into a single context string."""
        chunks = await self.retrieve(query, file_ids=file_ids, limit=limit)
        if not chunks:
            return ""
        parts = []
        for c in chunks:
            if include_sources:
                parts.append(
                    f"[Source: {c.file_id or 'unknown'}, Score: {c.score:.3f}]\n{c.text}"
                )
            else:
                parts.append(c.text)
        return separator.join(parts)

    async def aclose(self) -> None:
        if self.redis_client:
            await self.redis_client.aclose()
