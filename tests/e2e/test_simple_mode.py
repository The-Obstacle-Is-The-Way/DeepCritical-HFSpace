import pytest

from src.orchestrator import Orchestrator
from src.utils.models import OrchestratorConfig


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_simple_mode_completes(mock_search_handler, mock_judge_handler):
    """Verify Simple mode runs without crashing using mocks."""

    config = OrchestratorConfig(max_iterations=2)

    orchestrator = Orchestrator(
        search_handler=mock_search_handler,
        judge_handler=mock_judge_handler,
        config=config,
        enable_analysis=False,
        enable_embeddings=False,
    )

    events = []
    async for event in orchestrator.run("test query"):
        events.append(event)

    # Must complete
    assert any(e.type == "complete" for e in events), "Did not receive complete event"
    # Must not error
    assert not any(e.type == "error" for e in events), "Received error event"

    # Check structure of complete event
    complete_event = next(e for e in events if e.type == "complete")
    # The mock judge returns "MockDrug A" and "Finding 1", ensuring synthesis happens
    assert "MockDrug A" in complete_event.message
    assert "Finding 1" in complete_event.message


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_simple_mode_structure_validation(mock_search_handler, mock_judge_handler):
    """Verify output contains expected structure (citations, headings)."""
    config = OrchestratorConfig(max_iterations=2)
    orchestrator = Orchestrator(
        search_handler=mock_search_handler,
        judge_handler=mock_judge_handler,
        config=config,
        enable_analysis=False,
        enable_embeddings=False,
    )

    events = []
    async for event in orchestrator.run("test query"):
        events.append(event)

    complete_event = next(e for e in events if e.type == "complete")
    report = complete_event.message

    # Check markdown structure
    assert "## Drug Repurposing Analysis" in report
    assert "### Citations" in report
    assert "### Key Findings" in report

    # Check for citations
    assert "Study on test query" in report
    assert "https://pubmed.example.com/123" in report
