"""Unit tests for LangGraph state management."""

import operator

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph.state import Conflict, Hypothesis, ResearchState


def test_state_schema_definition():
    """Verify the ResearchState TypedDict structure."""
    # Just checking we can instantiate it (it's a TypedDict, so it's just a dict at runtime)
    state: ResearchState = {
        "query": "test query",
        "hypotheses": [],
        "conflicts": [],
        "evidence_ids": [],
        "messages": [],
        "next_step": "search",
        "iteration_count": 0,
        "max_iterations": 10,
    }
    assert state["query"] == "test query"
    assert state["next_step"] == "search"


def test_hypothesis_pydantic_model():
    """Verify Hypothesis Pydantic model validation."""
    hypo = Hypothesis(id="h1", statement="Test hypothesis", status="proposed", confidence=0.5)
    assert hypo.id == "h1"
    assert hypo.status == "proposed"
    assert hypo.confidence == 0.5
    # Test default lists
    assert hypo.supporting_evidence_ids == []


def test_state_reducers_simulation():
    """Simulate how LangGraph reduces state updates (operator.add)."""
    # Initial state
    messages = [HumanMessage(content="Start")]

    # Node 1 update (Search)
    new_messages = [AIMessage(content="Found results")]

    # Simulation of operator.add reducer
    messages = operator.add(messages, new_messages)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Found results"


def test_conflict_model():
    """Verify Conflict model."""
    conflict = Conflict(
        id="c1",
        description="Conflict A vs B",
        source_a_id="doc1",
        source_b_id="doc2",
        status="open",
    )
    assert conflict.status == "open"
    assert conflict.resolution is None
