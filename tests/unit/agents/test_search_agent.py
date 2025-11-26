"""Unit tests for SearchAgent."""

from unittest.mock import AsyncMock

import pytest

# Skip all tests if agent_framework not installed (optional dep)
pytest.importorskip("agent_framework")

from agent_framework import ChatMessage, Role

from src.agents.search_agent import SearchAgent
from src.utils.models import Citation, Evidence, SearchResult


@pytest.fixture
def mock_handler() -> AsyncMock:
    """Mock search handler."""
    handler = AsyncMock()
    handler.execute.return_value = SearchResult(
        query="test query",
        evidence=[
            Evidence(
                content="test content",
                citation=Citation(
                    source="pubmed",
                    title="Test Title",
                    url="http://test.com",
                    date="2023",
                    authors=["Author A"],
                ),
                relevance=1.0,
            )
        ],
        sources_searched=["pubmed"],
        total_found=1,
    )
    return handler


@pytest.mark.asyncio
async def test_run_executes_search(mock_handler: AsyncMock) -> None:
    """Test that run executes search and updates evidence store."""
    store: dict = {"current": []}
    agent = SearchAgent(mock_handler, store)

    response = await agent.run("test query")

    # Check handler called
    mock_handler.execute.assert_awaited_once_with("test query", max_results_per_tool=10)

    # Check store updated
    assert len(store["current"]) == 1
    assert store["current"][0].content == "test content"

    # Check response
    assert response.messages[0].role == Role.ASSISTANT
    assert "Found 1 sources" in response.messages[0].text


@pytest.mark.asyncio
async def test_run_handles_chat_message_input(mock_handler: AsyncMock) -> None:
    """Test that run handles ChatMessage input."""
    store: dict = {"current": []}
    agent = SearchAgent(mock_handler, store)

    message = ChatMessage(role=Role.USER, text="test query")
    await agent.run(message)

    mock_handler.execute.assert_awaited_once_with("test query", max_results_per_tool=10)


@pytest.mark.asyncio
async def test_run_handles_list_input(mock_handler: AsyncMock) -> None:
    """Test that run handles list of messages."""
    store: dict = {"current": []}
    agent = SearchAgent(mock_handler, store)

    messages = [
        ChatMessage(role=Role.SYSTEM, text="sys"),
        ChatMessage(role=Role.USER, text="test query"),
    ]
    await agent.run(messages)
    mock_handler.execute.assert_awaited_once_with("test query", max_results_per_tool=10)


@pytest.mark.asyncio
async def test_run_uses_embeddings(mock_handler: AsyncMock) -> None:
    """Test that run uses embedding service if provided."""
    store: dict = {"current": []}

    # Mock embedding service
    mock_embeddings = AsyncMock()
    # Mock deduplicate to return the evidence as is (or filtered)
    mock_embeddings.deduplicate.return_value = [
        Evidence(
            content="unique content",
            citation=Citation(source="pubmed", url="u1", title="t1", date="2024"),
        )
    ]
    # Mock search_similar to return related items
    mock_embeddings.search_similar.return_value = [
        {
            "id": "u2",
            "content": "related content",
            "metadata": {"source": "pubmed", "title": "related", "date": "2024"},
            "distance": 0.1,
        }
    ]

    agent = SearchAgent(mock_handler, store, embedding_service=mock_embeddings)

    await agent.run("test query")

    # Verify deduplicate called
    mock_embeddings.deduplicate.assert_awaited_once()

    # Verify semantic search called
    mock_embeddings.search_similar.assert_awaited_once_with("test query", n_results=5)

    # Verify store contains related evidence (if logic implemented to add it)
    # Note: logic for adding related evidence needs to be implemented in SearchAgent
    # The spec says: "Merge related evidence not already in results"

    # Check if u1 (deduplicated result) is in store
    assert any(e.citation.url == "u1" for e in store["current"])
    # Check if u2 (related result) is in store
    assert any(e.citation.url == "u2" for e in store["current"])
