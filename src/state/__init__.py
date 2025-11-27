"""State package - re-exports from agents.state for compatibility."""

from src.agents.state import (
    MagenticState,
    get_magentic_state,
    init_magentic_state,
)

__all__ = ["MagenticState", "get_magentic_state", "init_magentic_state"]
