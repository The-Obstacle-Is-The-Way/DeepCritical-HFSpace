"""Unit tests for graph nodes."""

import pytest

from src.agents.graph.nodes import judge_node, search_node, supervisor_node
from src.agents.graph.state import ResearchState


@pytest.mark.asyncio
async def test_judge_node_initialization(mocker):
    """Test judge creates initial hypothesis if none exist."""
    # Mock get_model to avoid needing real API keys
    mocker.patch("src.agents.graph.nodes.get_model", return_value=mocker.Mock())

    # Create a mock assessment with attributes
    mock_hypothesis = mocker.Mock()
    mock_hypothesis.drug = "Caffeine"
    mock_hypothesis.target = "Adenosine"
    mock_hypothesis.pathway = "CNS"
    mock_hypothesis.effect = "Alertness"
    mock_hypothesis.confidence = 0.8

    mock_assessment = mocker.Mock()
    mock_assessment.hypotheses = [mock_hypothesis]

    mock_result = mocker.Mock()
    mock_result.output = mock_assessment

    # Mock the Agent class entirely
    mock_agent_instance = mocker.Mock()
    mock_agent_instance.run = mocker.AsyncMock(return_value=mock_result)
    mocker.patch("src.agents.graph.nodes.Agent", return_value=mock_agent_instance)

    state: ResearchState = {
        "query": "Does coffee cause cancer?",
        "hypotheses": [],
        "conflicts": [],
        "evidence_ids": [],
        "messages": [],
        "next_step": "judge",
        "iteration_count": 0,
        "max_iterations": 10,
    }

    update = await judge_node(state)

    assert "hypotheses" in update
    assert len(update["hypotheses"]) == 1
    assert update["hypotheses"][0].id == "Caffeine"
    assert update["hypotheses"][0].status == "proposed"


@pytest.mark.asyncio
async def test_supervisor_termination():
    """Test supervisor forces synthesis at max iterations."""
    state: ResearchState = {
        "query": "test",
        "hypotheses": [],
        "conflicts": [],
        "evidence_ids": [],
        "messages": [],
        "next_step": "search",
        "iteration_count": 10,  # Max reached
        "max_iterations": 10,
    }

    update = await supervisor_node(state)
    assert update["next_step"] == "synthesize"


@pytest.mark.asyncio
async def test_search_node_execution(mocker):
    """Test search node calls tools (mocked)."""
    # Mock the tools
    mocker.patch("src.tools.pubmed.PubMedTool.search", return_value=[])
    mocker.patch("src.tools.clinicaltrials.ClinicalTrialsTool.search", return_value=[])
    mocker.patch("src.tools.europepmc.EuropePMCTool.search", return_value=[])

    state: ResearchState = {
        "query": "test",
        "hypotheses": [],
        "conflicts": [],
        "evidence_ids": [],
        "messages": [],
        "next_step": "search",
        "iteration_count": 0,
        "max_iterations": 10,
    }

    update = await search_node(state)
    assert "messages" in update
    # Matches "Found 0 total, 0 unique new papers."
    assert "0 unique new papers" in update["messages"][0].content
