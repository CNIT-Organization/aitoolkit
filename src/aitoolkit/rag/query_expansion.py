"""Query expansion for improved RAG recall.

Domain-agnostic: no domain-specific keyword list is hardcoded. Pass ``domain``
and an optional ``keywords`` list to tailor expansion to any project. With no
keywords, expansion is purely structural.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

from loguru import logger

# Generic English stop words for naive keyword extraction.
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "this", "that", "it",
}

# Content-type templates that anticipate document content for better retrieval.
_TEMPLATES = {
    "quiz": (
        "What are the key concepts, definitions, procedures, and important points "
        "related to {topic}? What questions should learners understand about {topic}?"
    ),
    "lesson": (
        "Explain {topic} in detail. What are the fundamental concepts of {topic}? "
        "What should learners understand about {topic}?"
    ),
    "summary": (
        "Summarize the main points about {topic}. What are the key takeaways "
        "regarding {topic}? Provide an overview of {topic}."
    ),
    "default": (
        "What topics and skills are needed to learn {topic}? "
        "Provide comprehensive material about {topic}."
    ),
}


class QueryExpander:
    """Expand short topics/questions into retrieval-optimized queries."""

    def __init__(
        self,
        domain: str = "general",
        keywords: Optional[List[str]] = None,
        domain_hint: Optional[str] = None,
    ) -> None:
        """
        Args:
            domain: Free-form domain label (used in chat expansion context).
            keywords: Optional domain keywords appended to keyword extraction and
                used to decide when to add ``domain_hint``.
            domain_hint: Extra phrase appended when a topic matches a keyword
                (e.g. "Include relevant regulations and safety procedures.").
        """
        self.domain = domain
        self.keywords = [k.lower() for k in (keywords or [])]
        self.domain_hint = domain_hint

    def expand_for_generation(
        self,
        topic: str,
        content_type: str = "default",
        file_names: Optional[List[str]] = None,
    ) -> str:
        """Expand a topic into a detailed query for content generation."""
        prefix = ""
        if file_names:
            clean = [_strip_ext(n) for n in file_names[:3]]
            if clean:
                prefix = f"Related to documents: {', '.join(clean)}. "

        template = _TEMPLATES.get(content_type, _TEMPLATES["default"])
        expanded = prefix + template.format(topic=topic)

        if self.domain_hint and self._matches_domain(topic):
            expanded += f" {self.domain_hint}"

        logger.debug(f"expanded '{topic}' -> '{expanded[:80]}...'")
        return expanded

    def expand_for_chat(self, question: str, conversation_history: str = "") -> str:
        """Lightly expand a short chat question for better retrieval."""
        if len(question.split()) >= 5:
            return question
        if conversation_history:
            return f"{question}. Context: {conversation_history[-200:]}"
        return f"{question} in {self.domain} context"

    def generate_query_variations(
        self, query: str, num_variations: int = 3
    ) -> List[str]:
        """Produce structural query variations for hybrid search."""
        variations = [query]
        lower = query.lower()
        if "what" not in lower and num_variations >= 2:
            variations.append(f"What is {query}? Explain {query}.")
        if "how" not in lower and num_variations >= 3:
            variations.append(f"How to {query}? Steps for {query}.")
        if self.keywords and num_variations >= 4:
            variations.append(f"{query} {' '.join(self.keywords[:5])}")
        return variations[:num_variations]

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Naive keyword extraction plus any configured domain keywords present."""
        words = [w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 3]
        domain_kws = [k for k in self.keywords if k in text.lower()]
        seen: List[str] = []
        for w in words + domain_kws:
            if w not in seen:
                seen.append(w)
        return seen[:top_n]

    def _matches_domain(self, topic: str) -> bool:
        if not self.keywords:
            return False
        low = topic.lower()
        return any(k in low for k in self.keywords)


def _strip_ext(name: str) -> str:
    for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".doc"):
        name = name.replace(ext, "")
    return name


@lru_cache(maxsize=16)
def _cached_expander(
    domain: str, keywords: Tuple[str, ...], domain_hint: Optional[str]
) -> QueryExpander:
    return QueryExpander(domain=domain, keywords=list(keywords), domain_hint=domain_hint)


def get_query_expander(
    domain: str = "general",
    keywords: Optional[List[str]] = None,
    domain_hint: Optional[str] = None,
) -> QueryExpander:
    """Return a cached query expander, one per (domain, keywords, hint) combo.

    Keying on the arguments avoids the single-global footgun where the first
    caller's domain/keywords would be returned for every subsequent call.
    """
    return _cached_expander(domain, tuple(keywords or ()), domain_hint)
