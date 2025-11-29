from unittest.mock import MagicMock

import pytest

from src.orchestrators.simple import Orchestrator
from src.utils.models import AssessmentDetails, JudgeAssessment


def make_assessment(
    mechanism: int,
    clinical: int,
    drug_candidates: list[str],
    sufficient: bool = False,
    recommendation: str = "continue",
    confidence: float = 0.8,
) -> JudgeAssessment:
    return JudgeAssessment(
        details=AssessmentDetails(
            mechanism_score=mechanism,
            mechanism_reasoning="reasoning is sufficient for testing purposes",
            clinical_evidence_score=clinical,
            clinical_reasoning="reasoning is sufficient for testing purposes",
            drug_candidates=drug_candidates,
            key_findings=["finding"],
        ),
        sufficient=sufficient,
        confidence=confidence,
        recommendation=recommendation,
        next_search_queries=[],
        reasoning="reasoning is sufficient for testing purposes",
    )


@pytest.fixture
def orchestrator():
    search = MagicMock()
    judge = MagicMock()
    return Orchestrator(search, judge)


def test_should_synthesize_high_scores(orchestrator):
    """High scores with drug candidates triggers synthesis."""
    assessment = make_assessment(mechanism=7, clinical=6, drug_candidates=["Metformin"])

    # Access the private method via name mangling or just call it if it was public.
    # Since I made it private _should_synthesize, I access it directly.
    should_synth, reason = orchestrator._should_synthesize(
        assessment, iteration=3, max_iterations=10, evidence_count=50
    )

    assert should_synth is True
    assert reason == "high_scores_with_candidates"


def test_should_synthesize_late_iteration(orchestrator):
    """Late iteration with acceptable scores triggers synthesis."""
    assessment = make_assessment(mechanism=5, clinical=4, drug_candidates=[])
    should_synth, reason = orchestrator._should_synthesize(
        assessment, iteration=9, max_iterations=10, evidence_count=80
    )

    assert should_synth is True
    assert reason in ["late_iteration_acceptable", "emergency_synthesis"]


def test_should_not_synthesize_early_low_scores(orchestrator):
    """Early iteration with low scores continues searching."""
    assessment = make_assessment(mechanism=3, clinical=2, drug_candidates=[])
    should_synth, reason = orchestrator._should_synthesize(
        assessment, iteration=2, max_iterations=10, evidence_count=20
    )

    assert should_synth is False
    assert reason == "continue_searching"


def test_judge_approved_overrides_all(orchestrator):
    """If judge explicitly says synthesize with good scores, do it."""
    assessment = make_assessment(
        mechanism=6, clinical=5, drug_candidates=[], sufficient=True, recommendation="synthesize"
    )
    should_synth, reason = orchestrator._should_synthesize(
        assessment, iteration=2, max_iterations=10, evidence_count=20
    )

    assert should_synth is True
    assert reason == "judge_approved"


def test_max_evidence_threshold(orchestrator):
    """Force synthesis if we have tons of evidence."""
    assessment = make_assessment(mechanism=2, clinical=2, drug_candidates=[])
    should_synth, reason = orchestrator._should_synthesize(
        assessment, iteration=5, max_iterations=10, evidence_count=150
    )

    assert should_synth is True
    assert reason == "max_evidence_reached"
