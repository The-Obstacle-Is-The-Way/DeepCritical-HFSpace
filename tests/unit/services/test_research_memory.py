"""Tests for the shared ResearchMemory service."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.agents.graph.state import Conflict, Hypothesis
from src.services.embedding_protocol import EmbeddingServiceProtocol
from src.services.research_memory import ResearchMemory
from src.utils.models import Citation, Evidence


@pytest.fixture
def mock_embedding_service():
    """Create a properly spec'd mock that matches EmbeddingServiceProtocol interface."""
    # Use create_autospec for proper interface enforcement
    service = create_autospec(EmbeddingServiceProtocol, instance=True)
    # Override with AsyncMock for async methods
    service.deduplicate = AsyncMock()
    service.add_evidence = AsyncMock()
    service.search_similar = AsyncMock()
    service.embed = AsyncMock()
    service.embed_batch = AsyncMock()
    return service


@pytest.fixture
def memory(mock_embedding_service):
    return ResearchMemory(query="test query", embedding_service=mock_embedding_service)


@pytest.mark.asyncio
async def test_store_evidence(memory, mock_embedding_service):
    # Setup
    ev1 = Evidence(
        content="content1",
        citation=Citation(source="pubmed", title="t1", url="u1", date="2023", authors=["a1"]),
    )
    ev2 = Evidence(
        content="content2",
        citation=Citation(source="pubmed", title="t2", url="u2", date="2023", authors=["a2"]),
    )

    # deduplicate returns only ev1 (simulating ev2 is duplicate)
    mock_embedding_service.deduplicate.return_value = [ev1]

    # Execute
    new_ids = await memory.store_evidence([ev1, ev2])

    # Verify
    assert new_ids == ["u1"]
    assert memory.evidence_ids == ["u1"]

    # deduplicate called with both (deduplicate() handles storage internally)
    mock_embedding_service.deduplicate.assert_called_once_with([ev1, ev2])

    # add_evidence should NOT be called separately (deduplicate() handles it)
    mock_embedding_service.add_evidence.assert_not_called()


@pytest.mark.asyncio
async def test_get_relevant_evidence(memory, mock_embedding_service):
    # Setup mock return from ChromaDB format
    mock_embedding_service.search_similar.return_value = [
        {
            "id": "u1",
            "content": "content1",
            "metadata": {
                "source": "pubmed",
                "title": "t1",
                "date": "2023",
                "authors": "a1,a2",
                "url": "u1",
            },
            "distance": 0.1,
        }
    ]

    # Execute
    results = await memory.get_relevant_evidence(n=5)

    # Verify
    assert len(results) == 1
    ev = results[0]
    assert isinstance(ev, Evidence)
    assert ev.content == "content1"
    assert ev.citation.title == "t1"
    assert ev.citation.authors == ["a1", "a2"]
    assert ev.relevance > 0.8  # 1.0 - 0.1 = 0.9


def test_hypothesis_tracking(memory):
    h1 = Hypothesis(id="h1", statement="drug -> target", status="confirmed", confidence=0.9)
    h2 = Hypothesis(id="h2", statement="drug -> unknown", status="proposed", confidence=0.5)

    memory.add_hypothesis(h1)
    memory.add_hypothesis(h2)

    assert len(memory.hypotheses) == 2
    confirmed = memory.get_confirmed_hypotheses()
    assert len(confirmed) == 1
    assert confirmed[0].id == "h1"


def test_conflict_tracking(memory):
    c1 = Conflict(id="c1", description="conflict", source_a_id="a", source_b_id="b", status="open")
    c2 = Conflict(
        id="c2",
        description="resolved conflict",
        source_a_id="a",
        source_b_id="b",
        status="resolved",
    )

    memory.add_conflict(c1)
    memory.add_conflict(c2)

    assert len(memory.conflicts) == 2
    open_conflicts = memory.get_open_conflicts()
    assert len(open_conflicts) == 1
    assert open_conflicts[0].id == "c1"
