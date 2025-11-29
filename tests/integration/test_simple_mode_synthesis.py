from unittest.mock import AsyncMock

import pytest

from src.orchestrators.simple import Orchestrator
from src.utils.models import (
    AssessmentDetails,
    Citation,
    Evidence,
    JudgeAssessment,
    OrchestratorConfig,
    SearchResult,
)


def make_evidence(title: str) -> Evidence:
    return Evidence(
        content="content",
        citation=Citation(title=title, url="http://test.com", date="2025", source="pubmed"),
    )


@pytest.mark.asyncio
async def test_simple_mode_synthesizes_before_max_iterations():
    """Verify simple mode produces useful output with mocked judge."""
    # Mock search to return evidence
    mock_search = AsyncMock()
    mock_search.execute.return_value = SearchResult(
        query="test query",
        evidence=[make_evidence(f"Paper {i}") for i in range(5)],
        errors=[],
        sources_searched=["pubmed"],
        total_found=5,
    )

    # Mock judge to return GOOD scores eventually
    # We can use MockJudgeHandler or a pure mock. Let's use a pure mock to control scores precisely.
    mock_judge = AsyncMock()

    # Iteration 1: Low scores
    assess_1 = JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=2,
            mechanism_reasoning="reasoning is sufficient for valid model",
            clinical_evidence_score=2,
            clinical_reasoning="reasoning is sufficient for valid model",
            drug_candidates=[],
            key_findings=[],
        ),
        sufficient=False,
        confidence=0.5,
        recommendation="continue",
        next_search_queries=["q2"],
        reasoning="need more evidence to support conclusions about this topic",
    )

    # Iteration 2: High scores (should trigger synthesis)
    assess_2 = JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=8,
            mechanism_reasoning="reasoning is sufficient for valid model",
            clinical_evidence_score=7,
            clinical_reasoning="reasoning is sufficient for valid model",
            drug_candidates=["MagicDrug"],
            key_findings=["It works"],
        ),
        sufficient=False,  # Judge is conservative
        confidence=0.9,
        recommendation="continue",  # Judge still says continue (simulating bias)
        next_search_queries=[],
        reasoning="good scores but maybe more evidence needed technically",
    )

    mock_judge.assess.side_effect = [assess_1, assess_2]

    orchestrator = Orchestrator(
        search_handler=mock_search,
        judge_handler=mock_judge,
        config=OrchestratorConfig(max_iterations=5),
    )

    events = []
    async for event in orchestrator.run("test query"):
        events.append(event)
        if event.type == "complete":
            break

    # Must have synthesis with drug candidates
    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) == 1
    complete_event = complete_events[0]

    assert "MagicDrug" in complete_event.message
    assert "Drug Candidates" in complete_event.message
    assert complete_event.data.get("synthesis_reason") == "high_scores_with_candidates"
    assert complete_event.iteration == 2  # Should stop at it 2


@pytest.mark.asyncio
async def test_partial_synthesis_generation():
    """Verify partial synthesis includes drug candidates even if max iterations reached."""
    mock_search = AsyncMock()
    mock_search.execute.return_value = SearchResult(
        query="test", evidence=[], errors=[], sources_searched=["pubmed"], total_found=0
    )

    mock_judge = AsyncMock()
    # Always return low scores but WITH candidates
    # Scores 3+3 = 6 < 8 (late threshold), so it should NOT synthesize early
    mock_judge.assess.return_value = JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=3,
            mechanism_reasoning="reasoning is sufficient for valid model",
            clinical_evidence_score=3,
            clinical_reasoning="reasoning is sufficient for valid model",
            drug_candidates=["PartialDrug"],
            key_findings=["Partial finding"],
        ),
        sufficient=False,
        confidence=0.5,
        recommendation="continue",
        next_search_queries=[],
        reasoning="keep going to find more evidence about this topic please",
    )

    orchestrator = Orchestrator(
        search_handler=mock_search,
        judge_handler=mock_judge,
        config=OrchestratorConfig(max_iterations=2),
    )

    events = []
    async for event in orchestrator.run("test"):
        events.append(event)

    complete_event = next(e for e in events if e.type == "complete")
    assert complete_event.data.get("max_reached") is True

    # The output message should contain the drug candidate from the last assessment
    assert "PartialDrug" in complete_event.message
    assert "Maximum iterations reached" in complete_event.message
