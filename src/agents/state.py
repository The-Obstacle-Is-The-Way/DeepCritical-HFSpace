"""Thread-safe state management for Magentic agents.

Uses contextvars to ensure isolation between concurrent requests (e.g., multiple users
searching simultaneously via Gradio).
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.services.research_memory import ResearchMemory

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class MagenticState(BaseModel):
    """Mutable state for a Magentic workflow session."""

    # We wrap ResearchMemory. Type as Any to avoid pydantic validation issues with complex objects
    memory: Any = None  # Instance of ResearchMemory

    model_config = {"arbitrary_types_allowed": True}


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
