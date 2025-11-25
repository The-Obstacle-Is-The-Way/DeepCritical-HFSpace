"""Unit tests for SearchAgent."""

from unittest.mock import AsyncMock

import pytest
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
