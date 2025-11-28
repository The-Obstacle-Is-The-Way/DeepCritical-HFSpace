"""Unit tests for hierarchical orchestration middleware."""

from unittest.mock import AsyncMock

import pytest

from src.middleware.sub_iteration import SubIterationMiddleware
from src.utils.models import AssessmentDetails, JudgeAssessment

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_sub_iteration_middleware():
    team = AsyncMock()
    team.execute.return_value = "Result"

    judge = AsyncMock()
    judge.assess.return_value = JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=10,
            mechanism_reasoning="Good reasoning text here",
            clinical_evidence_score=10,
            clinical_reasoning="Good reasoning text here",
            drug_candidates=[],
            key_findings=[],
        ),
        sufficient=True,
        confidence=1.0,
        recommendation="synthesize",
        next_search_queries=[],
        reasoning="Good reasoning text here for the overall assessment which must be long enough.",
    )

    middleware = SubIterationMiddleware(team, judge)
    result, assessment = await middleware.run("task")

    assert result == "Result"
    assert assessment.sufficient
    assert team.execute.call_count == 1
