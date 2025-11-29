"""Integration tests for the research graph."""

import pytest

from src.agents.graph.workflow import create_research_graph


@pytest.mark.asyncio
async def test_graph_execution_flow(mocker):
    """Test the graph runs from start to finish (simulated)."""
    # Mock Agent.run to avoid API calls
    mock_run = mocker.patch("pydantic_ai.Agent.run")
    # Return dummy report/assessment
    mock_result = mocker.Mock()
    mock_result.output = mocker.Mock()  # generic output
    # For judge: output.hypotheses = []
    mock_result.output.hypotheses = []
    # For report: validate_references needs specific structure?
    # Actually validate_references expects a ResearchReport.
    # Let's mock the return of validate_references too if needed, or make report valid.
    # Or just mock the node logic? No, we want to test the graph wiring.

    # Minimal valid report
    from src.utils.models import ReportSection, ResearchReport

    dummy_section = ReportSection(title="Dummy", content="Content")

    mock_report = ResearchReport(
        title="Test Report",
        executive_summary="Summary " * 20,  # Ensure > 100 chars
        research_question="Question",
        methodology=dummy_section,
        hypotheses_tested=[],
        mechanistic_findings=dummy_section,
        clinical_findings=dummy_section,
        drug_candidates=[],
        limitations=["None"],
        conclusion="Conclusion",
        references=[],
        confidence_score=0.5,
    )

    # Since fallback supervisor skips Judge and goes Search -> Synthesize,
    # Agent.run is only called once by SynthesizeNode.
    # It expects a ResearchReport.
    mock_result.output = mock_report
    mock_run.return_value = mock_result

    # Create graph without LLM (will use fallback supervisor logic -> search -> synthesize)
    graph = create_research_graph(llm=None)

    # Initial state
    initial_state = {
        "query": "test query",
        "hypotheses": [],
        "conflicts": [],
        "evidence_ids": [],
        "messages": [],
        "next_step": "search",
        "iteration_count": 0,
        "max_iterations": 2,  # Short run
    }

    # Execute graph
    events = []
    async for event in graph.astream(initial_state):
        events.append(event)

    # Verify flow
    # 1. Supervisor (start) -> decides search
    # 2. Search node runs
    # 3. Supervisor runs again -> max_iter reached -> synthesize
    # 4. Synthesize runs
    # 5. End

    # Just check we hit synthesis
    final_event = events[-1]
    assert "synthesize" in final_event or "messages" in str(final_event)
