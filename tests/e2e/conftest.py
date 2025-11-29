from unittest.mock import MagicMock

import pytest

from src.utils.models import AssessmentDetails, Citation, Evidence, JudgeAssessment, SearchResult


@pytest.fixture
def mock_search_handler():
    """Return a mock search handler that returns fake evidence."""
    mock = MagicMock()

    async def mock_execute(query, max_results=10):
        return SearchResult(
            query=query,
            evidence=[
                Evidence(
                    content=f"Evidence content for {query}",
                    citation=Citation(
                        source="pubmed",
                        title=f"Study on {query}",
                        url="https://pubmed.example.com/123",
                        date="2025-01-01",
                        authors=["Doe J"],
                    ),
                )
            ],
            sources_searched=["pubmed"],
            total_found=1,
            errors=[],
        )

    mock.execute = mock_execute
    return mock


@pytest.fixture
def mock_judge_handler():
    """Return a mock judge that always says 'synthesize'."""
    mock = MagicMock()

    async def mock_assess(question, evidence):
        return JudgeAssessment(
            sufficient=True,
            confidence=0.9,
            recommendation="synthesize",
            details=AssessmentDetails(
                mechanism_score=8,
                mechanism_reasoning="Strong mechanism found in mock data",
                clinical_evidence_score=7,
                clinical_reasoning="Good clinical evidence in mock data",
                drug_candidates=["MockDrug A"],
                key_findings=["Finding 1", "Finding 2"],
            ),
            reasoning="Evidence is sufficient for synthesis.",
            next_search_queries=[],
        )

    mock.assess = mock_assess
    return mock
