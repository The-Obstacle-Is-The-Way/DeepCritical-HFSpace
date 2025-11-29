"""Unit tests for ReportAgent."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if agent_framework not installed (optional dep)
pytest.importorskip("agent_framework")

from src.agents.report_agent import ReportAgent  # noqa: E402
from src.utils.models import (  # noqa: E402
    Citation,
    Evidence,
    MechanismHypothesis,
    ReportSection,
    ResearchReport,
)


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            content="Metformin activates AMPK...",
            citation=Citation(
                source="pubmed",
                title="Metformin mechanisms",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
                authors=["Smith J", "Jones A"],
            ),
        )
    ]


@pytest.fixture
def sample_hypotheses() -> list[MechanismHypothesis]:
    return [
        MechanismHypothesis(
            drug="Metformin",
            target="AMPK",
            pathway="mTOR inhibition",
            effect="Neuroprotection",
            confidence=0.8,
            search_suggestions=[],
        )
    ]


@pytest.fixture
def mock_report() -> ResearchReport:
    return ResearchReport(
        title="Drug Repurposing Analysis: Metformin for Alzheimer's",
        executive_summary=(
            "This report analyzes metformin as a potential candidate for "
            "repurposing in Alzheimer's disease treatment. It summarizes "
            "findings from mechanistic studies showing AMPK activation effects "
            "and reviews clinical data. The evidence suggests a potential "
            "neuroprotective role, although clinical trials are still limited."
        ),
        research_question="Can metformin be repurposed for Alzheimer's disease?",
        methodology=ReportSection(
            title="Methodology", content="Searched PubMed and web sources..."
        ),
        hypotheses_tested=[
            {"mechanism": "Metformin -> AMPK -> neuroprotection", "supported": 5, "contradicted": 1}
        ],
        mechanistic_findings=ReportSection(
            title="Mechanistic Findings", content="Evidence suggests AMPK activation..."
        ),
        clinical_findings=ReportSection(
            title="Clinical Findings", content="Limited clinical data available..."
        ),
        drug_candidates=["Metformin"],
        limitations=["Abstract-level analysis only"],
        conclusion="Metformin shows promise...",
        references=[],
        sources_searched=["pubmed", "web"],
        total_papers_reviewed=10,
        search_iterations=3,
        confidence_score=0.75,
    )


@pytest.mark.asyncio
async def test_report_agent_generates_report(
    sample_evidence: list[Evidence],
    sample_hypotheses: list[MechanismHypothesis],
    mock_report: ResearchReport,
) -> None:
    """ReportAgent should generate structured report."""
    store: dict[str, Any] = {
        "current": sample_evidence,
        "hypotheses": sample_hypotheses,
        "last_assessment": {"mechanism_score": 8, "clinical_score": 6},
    }

    with (
        patch("src.agents.report_agent.get_model") as mock_get_model,
        patch("src.agents.report_agent.Agent") as mock_agent_class,
    ):
        mock_get_model.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.output = mock_report
        mock_agent_class.return_value.run = AsyncMock(return_value=mock_result)

        agent = ReportAgent(store)
        response = await agent.run("metformin alzheimer")

        assert response.messages[0].text is not None
        assert "Executive Summary" in response.messages[0].text
        assert "Methodology" in response.messages[0].text
        assert "References" in response.messages[0].text
        # Verify report is stored in evidence store
        assert "final_report" in store


@pytest.mark.asyncio
async def test_report_agent_no_evidence() -> None:
    """ReportAgent should handle empty evidence gracefully."""
    store: dict[str, Any] = {"current": [], "hypotheses": []}

    # Lazy init means no patching needed - agent only instantiated when run() has evidence
    agent = ReportAgent(store)
    response = await agent.run("test query")

    assert response.messages[0].text is not None
    assert "Cannot generate report" in response.messages[0].text


# ═══════════════════════════════════════════════════════════════════════════
# 🚨 CRITICAL: Citation Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_report_agent_removes_hallucinated_citations(
    sample_evidence: list[Evidence],
) -> None:
    """ReportAgent should remove citations not in evidence."""
    from src.utils.citation_validator import validate_references

    # Create report with mix of valid and hallucinated references
    report_with_hallucinations = ResearchReport(
        title="Test Report",
        executive_summary=(
            "This is a test report for citation validation. It needs to be "
            "sufficiently long to pass validation. We are ensuring that the "
            "system correctly identifies and removes citations that do not "
            "appear in collected evidence. This prevents hallucinations."
        ),
        research_question="Testing citation validation",
        methodology=ReportSection(title="Methodology", content="Test"),
        hypotheses_tested=[],
        mechanistic_findings=ReportSection(title="Mechanistic", content="Test"),
        clinical_findings=ReportSection(title="Clinical", content="Test"),
        drug_candidates=["TestDrug"],
        limitations=["Test limitation"],
        conclusion="Test conclusion",
        references=[
            # Valid reference (matches sample_evidence)
            {
                "title": "Metformin mechanisms",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "authors": "Smith J, Jones A",
                "date": "2023",
                "source": "pubmed",
            },
            # HALLUCINATED reference (URL doesn't exist in evidence)
            {
                "title": "Fake Paper That Doesn't Exist",
                "url": "https://fake-journal.com/made-up-paper",
                "authors": "Hallucinated A",
                "date": "2024",
                "source": "fake",
            },
            # Another HALLUCINATED reference
            {
                "title": "Invented Research",
                "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/",
                "authors": "NotReal B",
                "date": "2025",
                "source": "pubmed",
            },
        ],
        sources_searched=["pubmed"],
        total_papers_reviewed=1,
        search_iterations=1,
        confidence_score=0.5,
    )

    # Validate - should remove hallucinated references
    validated_report = validate_references(report_with_hallucinations, sample_evidence)

    # Only the valid reference should remain
    assert len(validated_report.references) == 1
    assert validated_report.references[0]["title"] == "Metformin mechanisms"
    # Check that "Fake Paper" is NOT in the string representation of the references list
    # (This is a bit safer than checking presence in list of dicts if structure varies)
    ref_urls = [r.get("url") for r in validated_report.references]
    assert "https://fake-journal.com/made-up-paper" not in ref_urls


def test_citation_validator_handles_empty_references() -> None:
    """Citation validator should handle reports with no references."""
    from src.utils.citation_validator import validate_references

    report = ResearchReport(
        title="Empty Refs Report",
        executive_summary=(
            "This report has no references. It is designed to test the "
            "validator's handling of empty reference lists. We must ensure "
            "that the system does not crash when a report contains no "
            "citations. This is a valid edge case in early-stage research."
        ),
        research_question="Testing empty refs",
        methodology=ReportSection(title="Methodology", content="Test"),
        hypotheses_tested=[],
        mechanistic_findings=ReportSection(title="Mechanistic", content="Test"),
        clinical_findings=ReportSection(title="Clinical", content="Test"),
        drug_candidates=[],
        limitations=[],
        conclusion="Test",
        references=[],  # Empty!
        sources_searched=[],
        total_papers_reviewed=0,
        search_iterations=0,
        confidence_score=0.0,
    )

    validated = validate_references(report, [])
    assert validated.references == []
