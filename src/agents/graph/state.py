"""LangGraph state definitions for the research workflow."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# --- Domain Models (Inner Objects) ---
# We use Pydantic for strict validation of the data objects


class Hypothesis(BaseModel):
    """A research hypothesis with evidence tracking."""

    id: str = Field(description="Unique identifier for the hypothesis")
    statement: str = Field(description="The hypothesis statement")
    status: Literal["proposed", "validating", "confirmed", "refuted"] = Field(
        default="proposed", description="Current validation status"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str | None = Field(default=None, description="Reasoning for current status")


class Conflict(BaseModel):
    """A detected contradiction between sources."""

    id: str = Field(description="Unique identifier for the conflict")
    description: str = Field(description="Description of the contradiction")
    source_a_id: str = Field(description="ID of the first conflicting source")
    source_b_id: str = Field(description="ID of the second conflicting source")
    status: Literal["open", "resolved"] = Field(default="open")
    resolution: str | None = Field(default=None, description="Resolution explanation if resolved")


# --- Graph State (The Blackboard) ---
# LangGraph requires TypedDict for the main state object to support
# partial updates and reducers (operator.add).


class ResearchState(TypedDict):
    """The cognitive state shared across all graph nodes.

    Fields with 'Annotated[..., operator.add]' are reducers:
    returning a dict with these keys from a node will APPEND to the list
    instead of overwriting it.
    """

    # Immutable context
    query: str

    # Cognitive state (The "Blackboard")
    # Note: We store these as lists of Pydantic models.
    # Nodes should be careful to update existing items by ID if needed,
    # or we might need a custom reducer for merging by ID.
    # For now, we'll append and let the synthesizer filter the latest.
    hypotheses: Annotated[list[Hypothesis], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]

    # Evidence links (actual content stored in ChromaDB)
    evidence_ids: Annotated[list[str], operator.add]

    # Chat history (for LLM context)
    messages: Annotated[list[BaseMessage], operator.add]

    # Control flow
    next_step: Literal["search", "judge", "resolve", "synthesize", "finish"]
    iteration_count: int
    max_iterations: int
