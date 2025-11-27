"""Thread-safe state management for Magentic agents.

Uses contextvars to ensure isolation between concurrent requests (e.g., multiple users
searching simultaneously via Gradio).
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.utils.models import Citation, Evidence

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class MagenticState(BaseModel):
    """Mutable state for a Magentic workflow session."""

    evidence: list[Evidence] = Field(default_factory=list)
    # Type as Any to avoid circular imports/runtime resolution issues
    # The actual object injected will be an EmbeddingService instance
    embedding_service: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def add_evidence(self, new_evidence: list[Evidence]) -> int:
        """Add new evidence, deduplicating by URL.

        Returns:
            Number of *new* items added.
        """
        existing_urls = {e.citation.url for e in self.evidence}
        count = 0
        for item in new_evidence:
            if item.citation.url not in existing_urls:
                self.evidence.append(item)
                existing_urls.add(item.citation.url)
                count += 1
        return count

    async def search_related(self, query: str, n_results: int = 5) -> list[Evidence]:
        """Search for semantically related evidence using the embedding service."""
        if not self.embedding_service:
            return []

        results = await self.embedding_service.search_similar(query, n_results=n_results)

        # Convert dict results back to Evidence objects
        evidence_list = []
        for item in results:
            meta = item.get("metadata", {})
            authors_str = meta.get("authors", "")
            authors = [a.strip() for a in authors_str.split(",") if a.strip()]

            ev = Evidence(
                content=item["content"],
                citation=Citation(
                    title=meta.get("title", "Related Evidence"),
                    url=item["id"],
                    source="pubmed",  # Defaulting to pubmed if unknown
                    date=meta.get("date", "n.d."),
                    authors=authors,
                ),
                relevance=max(0.0, 1.0 - item.get("distance", 0.5)),
            )
            evidence_list.append(ev)

        return evidence_list


# The ContextVar holds the MagenticState for the current execution context
_magentic_state_var: ContextVar[MagenticState | None] = ContextVar("magentic_state", default=None)


def init_magentic_state(embedding_service: "EmbeddingService | None" = None) -> MagenticState:
    """Initialize a new state for the current context."""
    state = MagenticState(embedding_service=embedding_service)
    _magentic_state_var.set(state)
    return state


def get_magentic_state() -> MagenticState:
    """Get the current state. Raises RuntimeError if not initialized."""
    state = _magentic_state_var.get()
    if state is None:
        # Auto-initialize if missing (e.g. during tests or simple scripts)
        return init_magentic_state()
    return state
