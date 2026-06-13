"""Centralized RAG agent coordinating embeddings, vector store, and retrieval.

Provider-agnostic: no Google/Gemini defaults. ``answer_question`` is functional —
it retrieves context and generates an answer with the toolkit's
:class:`~aitoolkit.llm.LLMClient`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from aitoolkit.config import get_settings
from aitoolkit.embeddings import EmbeddingsClient, get_embeddings_client
from aitoolkit.llm import LLMClient, get_llm_client
from aitoolkit.rag.retriever import FilterFn, RAGRetriever
from aitoolkit.rag.vector_store import UnifiedVectorStore
from aitoolkit.types import RetrievedChunk

_DEFAULT_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the context is insufficient, say so clearly."
)


class RAGAgent:
    """Unified interface for embedding, storing, retrieving and answering."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        redis_url: Optional[str] = None,
        embeddings: Optional[EmbeddingsClient] = None,
        llm: Optional[LLMClient] = None,
        enable_caching: bool = True,
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        self.embeddings = embeddings or get_embeddings_client()
        self.llm = llm or get_llm_client()

        self.vector_store = UnifiedVectorStore(
            qdrant_url=qdrant_url,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            redis_url=(redis_url or settings.redis_url) if enable_caching else None,
            cache_ttl=settings.cache_ttl,
            filter_fn=filter_fn,
        )
        logger.success(f"RAGAgent ready (collection='{self.collection_name}')")

    async def add_documents(
        self,
        texts: List[str],
        *,
        file_id: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        source_type: str = "upload",
        **extra: Any,
    ) -> List[str]:
        """Embed and store document chunks. Returns point IDs."""
        return await self.vector_store.add_texts(
            texts, metadatas, file_id=file_id, source_type=source_type, **extra
        )

    async def delete_file(self, file_id: str) -> None:
        await self.vector_store.delete_by_file_id(file_id)

    async def retrieve_context(
        self,
        query: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[RetrievedChunk]:
        return await self.retriever.retrieve(
            query,
            file_ids=file_ids,
            limit=limit,
            score_threshold=score_threshold,
            use_cache=use_cache,
        )

    async def get_formatted_context(
        self,
        query: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        separator: str = "\n\n---\n\n",
    ) -> str:
        return await self.retriever.get_context_text(
            query, file_ids=file_ids, limit=limit, separator=separator
        )

    async def answer_question(
        self,
        question: str,
        *,
        file_ids: Optional[List[str]] = None,
        limit: int = 5,
        system: str = _DEFAULT_ANSWER_SYSTEM,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve context and generate a grounded answer."""
        chunks = await self.retrieve_context(
            question, file_ids=file_ids, limit=limit
        )
        if not chunks:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
            }

        context = "\n\n---\n\n".join(c.text for c in chunks)
        prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        answer = await self.llm.chat(
            prompt, system=system, temperature=temperature
        )
        return {
            "answer": answer,
            "sources": [c.model_dump() for c in chunks],
        }

    async def get_file_ids(self, exclude: Optional[List[str]] = None) -> List[str]:
        return await self.vector_store.get_unique_file_ids(exclude=exclude)

    async def get_file_count(self, exclude: Optional[List[str]] = None) -> int:
        return len(await self.get_file_ids(exclude=exclude))

    async def aclose(self) -> None:
        await self.retriever.aclose()
        await self.vector_store.aclose()


_agents: Dict[str, RAGAgent] = {}


def get_rag_agent(
    collection_name: Optional[str] = None,
    **kwargs: Any,
) -> RAGAgent:
    """Return a cached RAG agent, one per collection.

    Caching is keyed on the collection name so an app that serves multiple
    collections gets a distinct agent for each — a single global singleton would
    silently hand back the first collection's agent for every later call.
    ``kwargs`` only take effect the first time a given collection is requested.
    """
    key = collection_name or get_settings().qdrant_collection
    agent = _agents.get(key)
    if agent is None:
        agent = RAGAgent(collection_name=collection_name, **kwargs)
        _agents[key] = agent
    return agent
