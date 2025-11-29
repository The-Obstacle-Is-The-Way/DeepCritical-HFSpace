"""Unit tests for HypothesisAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if agent_framework not installed (optional dep)
pytest.importorskip("agent_framework")

from agent_framework import AgentRunResponse  # noqa: E402

from src.agents.hypothesis_agent import HypothesisAgent  # noqa: E402
from src.utils.models import (  # noqa: E402
    Citation,
    Evidence,
    HypothesisAssessment,
    MechanismHypothesis,
)


@pytest.fixture
def sample_evidence():
    return [
        Evidence(
            content="Metformin activates AMPK, which inhibits mTOR signaling...",
            citation=Citation(
                source="pubmed",
                title="Metformin and AMPK",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
            ),
        )
    ]


@pytest.fixture
def mock_assessment():
    return HypothesisAssessment(
        hypotheses=[
            MechanismHypothesis(
                drug="Metformin",
                target="AMPK",
                pathway="mTOR inhibition",
                effect="Reduced cancer cell proliferation",
                confidence=0.75,
                search_suggestions=["metformin AMPK cancer", "mTOR cancer therapy"],
            )
        ],
        primary_hypothesis=None,
        knowledge_gaps=["Clinical trial data needed"],
        recommended_searches=["metformin clinical trial cancer"],
    )


@pytest.mark.asyncio
async def test_hypothesis_agent_generates_hypotheses(sample_evidence, mock_assessment):
    """HypothesisAgent should generate mechanistic hypotheses."""
    store = {"current": sample_evidence, "hypotheses": []}

    with patch("src.agents.hypothesis_agent.get_model") as mock_get_model:
        with patch("src.agents.hypothesis_agent.Agent") as mock_agent_class:
            mock_get_model.return_value = MagicMock()  # Mock model
            mock_result = MagicMock()
            mock_result.output = mock_assessment
            # pydantic-ai Agent returns an object with .output for structured output
            mock_agent_class.return_value.run = AsyncMock(return_value=mock_result)

            agent = HypothesisAgent(store)
            response = await agent.run("metformin cancer")

            assert isinstance(response, AgentRunResponse)
            assert "AMPK" in response.messages[0].text
            assert len(store["hypotheses"]) == 1
            assert store["hypotheses"][0].drug == "Metformin"


@pytest.mark.asyncio
async def test_hypothesis_agent_no_evidence():
    """HypothesisAgent should handle empty evidence gracefully."""
    store = {"current": [], "hypotheses": []}

    # No need to mock Agent/get_model - empty evidence returns early
    agent = HypothesisAgent(store)
    response = await agent.run("test query")

    assert "No evidence" in response.messages[0].text
    assert len(store["hypotheses"]) == 0


@pytest.mark.asyncio
async def test_hypothesis_agent_uses_embeddings(sample_evidence, mock_assessment):
    """HypothesisAgent should pass embeddings to prompt formatter."""
    store = {"current": sample_evidence, "hypotheses": []}
    mock_embeddings = MagicMock()

    with patch("src.agents.hypothesis_agent.get_model") as mock_get_model:
        with patch("src.agents.hypothesis_agent.Agent") as mock_agent_class:
            # Mock format_hypothesis_prompt to check if embeddings were passed
            with patch("src.agents.hypothesis_agent.format_hypothesis_prompt") as mock_format:
                mock_get_model.return_value = MagicMock()  # Mock model
                mock_format.return_value = "Prompt"

                mock_result = MagicMock()
                mock_result.output = mock_assessment
                mock_agent_class.return_value.run = AsyncMock(return_value=mock_result)

                agent = HypothesisAgent(store, embedding_service=mock_embeddings)
                await agent.run("query")

                mock_format.assert_called_once()
                _args, kwargs = mock_format.call_args
                assert kwargs["embeddings"] == mock_embeddings
                assert _args[0] == "query"  # First positional arg is query
                assert _args[1] == sample_evidence  # Second positional arg is evidence
