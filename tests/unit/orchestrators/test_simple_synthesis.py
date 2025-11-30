"""Tests for simple orchestrator LLM synthesis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrators.simple import Orchestrator
from src.utils.models import AssessmentDetails, Citation, Evidence, JudgeAssessment


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Sample evidence for testing synthesis."""
    return [
        Evidence(
            content="Testosterone therapy demonstrates efficacy in treating HSDD.",
            citation=Citation(
                source="pubmed",
                title="Testosterone and Female Sexual Desire",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
                authors=["Smith J", "Jones A"],
            ),
        ),
        Evidence(
            content="A meta-analysis of 8 RCTs shows significant improvement in sexual desire.",
            citation=Citation(
                source="pubmed",
                title="Meta-analysis of Testosterone Therapy",
                url="https://pubmed.ncbi.nlm.nih.gov/67890/",
                date="2024",
                authors=["Johnson B"],
            ),
        ),
    ]


@pytest.fixture
def sample_assessment() -> JudgeAssessment:
    """Sample assessment for testing synthesis."""
    return JudgeAssessment(
        sufficient=True,
        confidence=0.85,
        reasoning="Evidence is sufficient to synthesize findings on testosterone therapy for HSDD.",
        recommendation="synthesize",
        next_search_queries=[],
        details=AssessmentDetails(
            mechanism_score=8,
            mechanism_reasoning="Strong evidence of androgen receptor activation pathway.",
            clinical_evidence_score=7,
            clinical_reasoning="Multiple RCTs support efficacy in postmenopausal HSDD.",
            drug_candidates=["Testosterone", "LibiGel"],
            key_findings=[
                "Testosterone improves libido in postmenopausal women",
                "Transdermal formulation has best safety profile",
            ],
        ),
    )


@pytest.mark.unit
class TestGenerateSynthesis:
    """Tests for _generate_synthesis method."""

    @pytest.mark.asyncio
    async def test_calls_llm_for_narrative(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Synthesis should make an LLM call, not just use a template."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]  # Needed for footer

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch("src.agent_factory.judges.get_model") as mock_get_model,
        ):
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.output = """### Executive Summary

Testosterone therapy demonstrates consistent efficacy for HSDD treatment.

### Background

HSDD affects many postmenopausal women.

### Evidence Synthesis

Studies show significant improvement in sexual desire scores.

### Recommendations

1. Consider testosterone therapy for postmenopausal HSDD

### Limitations

Long-term safety data is limited.

### References

1. Smith J et al. (2023). Testosterone and Female Sexual Desire."""

            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await orchestrator._generate_synthesis(
                query="testosterone HSDD",
                evidence=sample_evidence,
                assessment=sample_assessment,
            )

            # Verify LLM agent was created and called
            mock_agent_class.assert_called_once()
            mock_agent.run.assert_called_once()

            # Verify output includes narrative content
            assert "Executive Summary" in result
            assert "Background" in result
            assert "Evidence Synthesis" in result

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error_with_notice(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Synthesis should fall back to template if LLM fails, WITH error notice."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]

        with patch("pydantic_ai.Agent") as mock_agent_class:
            # Simulate LLM failure
            mock_agent_class.side_effect = Exception("LLM unavailable")

            result = await orchestrator._generate_synthesis(
                query="testosterone HSDD",
                evidence=sample_evidence,
                assessment=sample_assessment,
            )

            # Should surface error to user (MS Agent Framework pattern)
            assert "AI narrative synthesis unavailable" in result
            assert "Error" in result

            # Should still include template content
            assert "Assessment" in result or "Drug Candidates" in result
            assert "Testosterone" in result  # Drug candidate should be present

    @pytest.mark.asyncio
    async def test_includes_citation_footer(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Synthesis should include full citation list footer."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]

        with (
            patch("pydantic_ai.Agent") as mock_agent_class,
            patch("src.agent_factory.judges.get_model"),
        ):
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.output = "Narrative synthesis content."
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await orchestrator._generate_synthesis(
                query="test query",
                evidence=sample_evidence,
                assessment=sample_assessment,
            )

            # Should include citation footer
            assert "Full Citation List" in result
            assert "pubmed.ncbi.nlm.nih.gov/12345" in result
            assert "pubmed.ncbi.nlm.nih.gov/67890" in result


@pytest.mark.unit
class TestGenerateTemplateSynthesis:
    """Tests for _generate_template_synthesis fallback method."""

    def test_returns_structured_output(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Template synthesis should return structured markdown."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]

        result = orchestrator._generate_template_synthesis(
            query="testosterone HSDD",
            evidence=sample_evidence,
            assessment=sample_assessment,
        )

        # Should have all required sections
        assert "Question" in result
        assert "Drug Candidates" in result
        assert "Key Findings" in result
        assert "Assessment" in result
        assert "Citations" in result

    def test_includes_drug_candidates(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Template synthesis should list drug candidates."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]

        result = orchestrator._generate_template_synthesis(
            query="test",
            evidence=sample_evidence,
            assessment=sample_assessment,
        )

        assert "Testosterone" in result
        assert "LibiGel" in result

    def test_includes_scores(
        self,
        sample_evidence: list[Evidence],
        sample_assessment: JudgeAssessment,
    ) -> None:
        """Template synthesis should include assessment scores."""
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )
        orchestrator.history = [{"iteration": 1}]

        result = orchestrator._generate_template_synthesis(
            query="test",
            evidence=sample_evidence,
            assessment=sample_assessment,
        )

        assert "8/10" in result  # Mechanism score
        assert "7/10" in result  # Clinical score
        assert "85%" in result  # Confidence
