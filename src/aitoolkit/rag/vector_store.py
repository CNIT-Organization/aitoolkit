"""Unified Qdrant vector store with file-based filtering.

LangChain-free: it works with plain ``texts`` + ``metadatas`` and returns
:class:`~aitoolkit.types.RetrievedChunk` objects. Embeddings are produced by an
injected :class:`~aitoolkit.embeddings.EmbeddingsClient`, so the vector size is
detected at runtime rather than hardcoded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from aitoolkit.config import get_settings
from aitoolkit.embeddings import EmbeddingsClient, get_embeddings_client
from aitoolkit.exceptions import VectorStoreError
from aitoolkit.types import RetrievedChunk


class UnifiedVectorStore:
    """Async Qdrant wrapper for storing and retrieving document embeddings."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[EmbeddingsClient] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection
        self.embeddings = embeddings or get_embeddings_client()
        self._client = AsyncQdrantClient(
            url=self.qdrant_url,
            check_compatibility=settings.qdrant_check_compatibility,
        )
        self._initialized = False
        self._vector_size: Optional[int] = vector_size or settings.qdrant_vector_size
        logger.info(
            f"UnifiedVectorStore ready (collection='{self.collection_name}', "
            f"url={self.qdrant_url})"
        )

    async def _ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """Create the collection if it does not exist (no destructive recreate)."""
        if self._initialized:
            return

        try:
            exists = await self._client.collection_exists(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to check collection: {exc}") from exc

        if exists:
            self._initialized = True
            try:
                info = await self._client.get_collection(self.collection_name)
                size = info.config.params.vectors.size  # type: ignore[union-attr]
                if isinstance(size, int):
                    self._vector_size = size
            except Exception:  # noqa: BLE001 - size discovery is best-effort
                pass
            return

        size = vector_size or self._vector_size
        if size is None:
            # Nothing to create yet — caller will provide a size once embeddings exist.
            return

        try:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=size, distance=qmodels.Distance.COSINE
                ),
            )
            self._initialized = True
            self._vector_size = size
            logger.success(
                f"Created collection '{self.collection_name}' (size={size})"
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to create collection: {exc}") from exc

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        file_id: str,
        source_type: str = "upload",
        **extra_metadata: Any,
    ) -> List[str]:
        """Embed and store ``texts`` with associated metadata. Returns point IDs."""
        if not texts:
            return []

        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise VectorStoreError("texts and metadatas length mismatch")

        embeddings = await self.embeddings.aembed_documents(texts)
        if not embeddings:
            logger.warning("No embeddings produced; nothing stored")
            return []

        await self._ensure_collection(len(embeddings[0]))

        point_ids: List[str] = []
        points: List[qmodels.PointStruct] = []
        for text, vector, meta in zip(texts, embeddings, metadatas):
            pid = str(uuid4())
            point_ids.append(pid)
            points.append(
                qmodels.PointStruct(
                    id=pid,
                    vector=vector,
                    payload={
                        "text": text,
                        "file_id": file_id,
                        "source_type": source_type,
                        **meta,
                        **extra_metadata,
                    },
                )
            )

        try:
            await self._client.upsert(
                collection_name=self.collection_name, points=points
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to upsert points: {exc}") from exc

        logger.info(f"Stored {len(points)} chunks for file_id={file_id}")
        return point_ids

    async def delete_by_file_id(self, file_id: str) -> None:
        """Delete all points for a given ``file_id``."""
        await self._ensure_collection()
        if not self._initialized:
            return
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="file_id",
                                match=qmodels.MatchValue(value=file_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted points for file_id={file_id}")
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to delete by file_id: {exc}") from exc

    async def similarity_search(
        self,
        query: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Vector similarity search with optional ``file_ids`` filtering."""
        query_vector = await self.embeddings.aembed_query(query)
        await self._ensure_collection(len(query_vector))
        if not self._initialized:
            return []

        qfilter = None
        if file_ids:
            qfilter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="file_id", match=qmodels.MatchAny(any=file_ids)
                    )
                ]
            )

        try:
            response = await self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qfilter,
                score_threshold=score_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"similarity search failed: {exc}") from exc

        chunks: List[RetrievedChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunks.append(
                RetrievedChunk(
                    text=text,
                    score=point.score,
                    file_id=payload.get("file_id"),
                    metadata=payload,
                )
            )
        logger.debug(f"similarity_search returned {len(chunks)} chunks")
        return chunks

    async def get_unique_file_ids(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Return all distinct ``file_id`` values stored in the collection."""
        await self._ensure_collection()
        if not self._initialized:
            return []
        exclude_set = set(exclude or [])
        found: set[str] = set()
        offset = None
        try:
            while True:
                records, offset = await self._client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for rec in records:
                    fid = (rec.payload or {}).get("file_id")
                    if fid and fid not in exclude_set:
                        found.add(fid)
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.error(f"failed to scroll file_ids: {exc}")
            return []
        return sorted(found)

    async def aclose(self) -> None:
        await self._client.close()
