"""Thread-safe state management for Magentic agents.

Uses contextvars to ensure isolation between concurrent requests (e.g., multiple users
searching simultaneously via Gradio).
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from src.services.research_memory import ResearchMemory

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService
    from src.utils.models import Evidence


class MagenticState(BaseModel):
    """Mutable state for a Magentic workflow session."""

    # We wrap ResearchMemory. Type as Any to avoid pydantic validation issues with complex objects
    memory: Any = None  # Instance of ResearchMemory

    model_config = {"arbitrary_types_allowed": True}

    # --- Proxy methods for backwards compatibility with retrieval_agent.py ---

    async def add_evidence(self, evidence: list["Evidence"]) -> int:
        """Add evidence to memory with deduplication and embedding storage.

        This method delegates to ResearchMemory.store_evidence() which:
        1. Performs semantic deduplication (threshold 0.9)
        2. Stores unique evidence in the vector store
        3. Caches evidence for retrieval

        Args:
            evidence: List of Evidence objects to store.

        Returns:
            Number of new (non-duplicate) evidence items stored.
        """
        if self.memory is None:
            return 0

        memory: ResearchMemory = self.memory
        initial_count = len(memory.evidence_ids)
        await memory.store_evidence(evidence)
        return len(memory.evidence_ids) - initial_count

    @property
    def embedding_service(self) -> "EmbeddingService | None":
        """Get the embedding service from memory."""
        if self.memory is None:
            return None
        # Cast needed because memory is typed as Any to avoid Pydantic issues
        from src.services.embeddings import EmbeddingService as EmbeddingSvc

        return cast(EmbeddingSvc | None, self.memory._embedding_service)


# The ContextVar holds the MagenticState for the current execution context
_magentic_state_var: ContextVar[MagenticState | None] = ContextVar("magentic_state", default=None)


def init_magentic_state(
    query: str, embedding_service: "EmbeddingService | None" = None
) -> MagenticState:
    """Initialize a new state for the current context."""
    memory = ResearchMemory(query=query, embedding_service=embedding_service)
    state = MagenticState(memory=memory)
    _magentic_state_var.set(state)
    return state


def get_magentic_state() -> MagenticState:
    """Get the current state. Raises RuntimeError if not initialized."""
    state = _magentic_state_var.get()
    if state is None:
        raise RuntimeError("MagenticState not initialized. Call init_magentic_state() first.")
    return state
