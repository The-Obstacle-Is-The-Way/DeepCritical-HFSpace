"""Unit tests for JudgeAgent."""

from unittest.mock import AsyncMock

import pytest

# Skip all tests if agent_framework not installed (optional dep)
pytest.importorskip("agent_framework")

from agent_framework import ChatMessage, Role  # noqa: E402

from src.agents.judge_agent import JudgeAgent  # noqa: E402
from src.utils.models import AssessmentDetails, Citation, Evidence, JudgeAssessment  # noqa: E402


@pytest.fixture
def mock_assessment() -> JudgeAssessment:
    """Create a mock JudgeAssessment."""
    return JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=8,
            mechanism_reasoning="Strong mechanism evidence",
            clinical_evidence_score=7,
            clinical_reasoning="Good clinical data",
            drug_candidates=["Metformin"],
            key_findings=["Key finding 1"],
        ),
        sufficient=True,
        confidence=0.85,
        recommendation="synthesize",
        next_search_queries=[],
        reasoning="Evidence is sufficient for synthesis",
    )


@pytest.fixture
def mock_handler(mock_assessment: JudgeAssessment) -> AsyncMock:
    """Mock judge handler."""
    handler = AsyncMock()
    handler.assess.return_value = mock_assessment
    return handler


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Sample evidence for tests."""
    return [
        Evidence(
            content="Test content",
            citation=Citation(
                source="pubmed",
                title="Test Title",
                url="http://test.com",
                date="2023",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_run_assesses_evidence(
    mock_handler: AsyncMock,
    sample_evidence: list[Evidence],
) -> None:
    """Test that run assesses evidence from store."""
    store: dict = {"current": sample_evidence}
    agent = JudgeAgent(mock_handler, store)

    response = await agent.run("test question")

    # Check handler called with evidence from store
    mock_handler.assess.assert_awaited_once()
    call_args = mock_handler.assess.call_args
    assert call_args[0][0] == "test question"
    assert call_args[0][1] == sample_evidence

    # Check response
    assert response.messages[0].role == Role.ASSISTANT
    assert "synthesize" in response.messages[0].text


@pytest.mark.asyncio
async def test_run_handles_chat_message_input(
    mock_handler: AsyncMock,
    sample_evidence: list[Evidence],
) -> None:
    """Test that run handles ChatMessage input."""
    store: dict = {"current": sample_evidence}
    agent = JudgeAgent(mock_handler, store)

    message = ChatMessage(role=Role.USER, text="test question")
    await agent.run(message)

    mock_handler.assess.assert_awaited_once()
    assert mock_handler.assess.call_args[0][0] == "test question"


@pytest.mark.asyncio
async def test_run_handles_list_input(
    mock_handler: AsyncMock,
    sample_evidence: list[Evidence],
) -> None:
    """Test that run handles list of messages."""
    store: dict = {"current": sample_evidence}
    agent = JudgeAgent(mock_handler, store)

    messages = [
        ChatMessage(role=Role.SYSTEM, text="sys"),
        ChatMessage(role=Role.USER, text="test question"),
    ]
    await agent.run(messages)

    mock_handler.assess.assert_awaited_once()
    assert mock_handler.assess.call_args[0][0] == "test question"


@pytest.mark.asyncio
async def test_run_uses_empty_evidence_when_store_empty(
    mock_handler: AsyncMock,
) -> None:
    """Test that run works with empty evidence store."""
    store: dict = {"current": []}
    agent = JudgeAgent(mock_handler, store)

    await agent.run("test")

    mock_handler.assess.assert_awaited_once()
    assert mock_handler.assess.call_args[0][1] == []
