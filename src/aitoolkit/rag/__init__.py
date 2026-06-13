"""RAG capability (optional extra ``aitoolkit[rag]``).

Imports require ``qdrant-client``. Install with ``pip install 'aitoolkit[rag]'``.
"""

try:  # pragma: no cover - import guard
    import qdrant_client  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The RAG module requires qdrant-client. Install with: pip install 'aitoolkit[rag]'"
    ) from exc

from aitoolkit.rag.agent import RAGAgent, get_rag_agent
from aitoolkit.rag.query_expansion import QueryExpander, get_query_expander
from aitoolkit.rag.retriever import RAGRetriever
from aitoolkit.rag.vector_store import UnifiedVectorStore

__all__ = [
    "RAGAgent",
    "get_rag_agent",
    "RAGRetriever",
    "UnifiedVectorStore",
    "QueryExpander",
    "get_query_expander",
]
