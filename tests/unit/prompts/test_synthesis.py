"""Tests for narrative synthesis prompts."""

import pytest

from src.prompts.synthesis import (
    FEW_SHOT_EXAMPLE,
    format_synthesis_prompt,
    get_synthesis_system_prompt,
)


@pytest.mark.unit
class TestSynthesisSystemPrompt:
    """Tests for synthesis system prompt generation."""

    def test_system_prompt_emphasizes_prose(self) -> None:
        """System prompt should emphasize prose paragraphs, not bullets."""
        prompt = get_synthesis_system_prompt()
        assert "PROSE PARAGRAPHS" in prompt
        assert "not bullet points" in prompt.lower()

    def test_system_prompt_requires_executive_summary(self) -> None:
        """System prompt should require executive summary section."""
        prompt = get_synthesis_system_prompt()
        assert "Executive Summary" in prompt
        assert "REQUIRED" in prompt

    def test_system_prompt_requires_background(self) -> None:
        """System prompt should require background section."""
        prompt = get_synthesis_system_prompt()
        assert "Background" in prompt

    def test_system_prompt_requires_evidence_synthesis(self) -> None:
        """System prompt should require evidence synthesis section."""
        prompt = get_synthesis_system_prompt()
        assert "Evidence Synthesis" in prompt
        assert "Mechanism of Action" in prompt

    def test_system_prompt_requires_recommendations(self) -> None:
        """System prompt should require recommendations section."""
        prompt = get_synthesis_system_prompt()
        assert "Recommendations" in prompt

    def test_system_prompt_requires_limitations(self) -> None:
        """System prompt should require limitations section."""
        prompt = get_synthesis_system_prompt()
        assert "Limitations" in prompt

    def test_system_prompt_warns_about_hallucination(self) -> None:
        """System prompt should warn about citation hallucination."""
        prompt = get_synthesis_system_prompt()
        assert "NEVER hallucinate" in prompt or "never hallucinate" in prompt.lower()

    def test_system_prompt_includes_domain_name(self) -> None:
        """System prompt should include domain name."""
        prompt = get_synthesis_system_prompt("sexual_health")
        assert "sexual health" in prompt.lower()


@pytest.mark.unit
class TestFormatSynthesisPrompt:
    """Tests for synthesis user prompt formatting."""

    def test_includes_query(self) -> None:
        """User prompt should include the research query."""
        prompt = format_synthesis_prompt(
            query="testosterone libido",
            evidence_summary="Study shows efficacy...",
            drug_candidates=["Testosterone"],
            key_findings=["Improved libido"],
            mechanism_score=8,
            clinical_score=7,
            confidence=0.85,
        )
        assert "testosterone libido" in prompt

    def test_includes_evidence_summary(self) -> None:
        """User prompt should include evidence summary."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="Study by Smith et al. shows significant results...",
            drug_candidates=[],
            key_findings=[],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "Study by Smith et al." in prompt

    def test_includes_drug_candidates(self) -> None:
        """User prompt should include drug candidates."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=["Testosterone", "Flibanserin"],
            key_findings=[],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "Testosterone" in prompt
        assert "Flibanserin" in prompt

    def test_includes_key_findings(self) -> None:
        """User prompt should include key findings."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=[],
            key_findings=["Improved libido in postmenopausal women", "Safe profile"],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "Improved libido in postmenopausal women" in prompt
        assert "Safe profile" in prompt

    def test_includes_scores(self) -> None:
        """User prompt should include assessment scores."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=[],
            key_findings=[],
            mechanism_score=8,
            clinical_score=7,
            confidence=0.85,
        )
        assert "8/10" in prompt
        assert "7/10" in prompt
        assert "85%" in prompt

    def test_handles_empty_candidates(self) -> None:
        """User prompt should handle empty drug candidates."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=[],
            key_findings=[],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "None identified" in prompt

    def test_handles_empty_findings(self) -> None:
        """User prompt should handle empty key findings."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=[],
            key_findings=[],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "No specific findings" in prompt

    def test_includes_few_shot_example(self) -> None:
        """User prompt should include few-shot example."""
        prompt = format_synthesis_prompt(
            query="test query",
            evidence_summary="...",
            drug_candidates=[],
            key_findings=[],
            mechanism_score=5,
            clinical_score=5,
            confidence=0.5,
        )
        assert "Alprostadil" in prompt  # From the few-shot example


@pytest.mark.unit
class TestFewShotExample:
    """Tests for the few-shot example quality."""

    def test_few_shot_is_mostly_narrative(self) -> None:
        """Few-shot example should be mostly prose paragraphs, not bullets."""
        # Count substantial paragraphs (>100 chars of prose)
        paragraphs = [p for p in FEW_SHOT_EXAMPLE.split("\n\n") if len(p) > 100]
        # Count bullet points
        bullets = FEW_SHOT_EXAMPLE.count("\n- ") + FEW_SHOT_EXAMPLE.count("\n1. ")

        # Prose should dominate - at least as many paragraphs as bullets
        assert len(paragraphs) >= bullets, "Few-shot example should be mostly narrative prose"

    def test_few_shot_has_executive_summary(self) -> None:
        """Few-shot example should demonstrate executive summary."""
        assert "Executive Summary" in FEW_SHOT_EXAMPLE

    def test_few_shot_has_background(self) -> None:
        """Few-shot example should demonstrate background section."""
        assert "Background" in FEW_SHOT_EXAMPLE

    def test_few_shot_has_evidence_synthesis(self) -> None:
        """Few-shot example should demonstrate evidence synthesis."""
        assert "Evidence Synthesis" in FEW_SHOT_EXAMPLE
        assert "Mechanism of Action" in FEW_SHOT_EXAMPLE

    def test_few_shot_has_recommendations(self) -> None:
        """Few-shot example should demonstrate recommendations."""
        assert "Recommendations" in FEW_SHOT_EXAMPLE

    def test_few_shot_has_limitations(self) -> None:
        """Few-shot example should demonstrate limitations."""
        assert "Limitations" in FEW_SHOT_EXAMPLE

    def test_few_shot_has_references(self) -> None:
        """Few-shot example should demonstrate references format."""
        assert "References" in FEW_SHOT_EXAMPLE
        assert "pubmed.ncbi.nlm.nih.gov" in FEW_SHOT_EXAMPLE

    def test_few_shot_includes_statistics(self) -> None:
        """Few-shot example should demonstrate statistical reporting."""
        assert "%" in FEW_SHOT_EXAMPLE  # Percentages
        assert "p<" in FEW_SHOT_EXAMPLE or "p=" in FEW_SHOT_EXAMPLE  # P-values
        assert "CI" in FEW_SHOT_EXAMPLE  # Confidence intervals
