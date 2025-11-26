"""Judge handler for evidence assessment using PydanticAI."""

from typing import Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.prompts.judge import (
    SYSTEM_PROMPT,
    format_empty_evidence_prompt,
    format_user_prompt,
)
from src.utils.config import settings
from src.utils.models import AssessmentDetails, Evidence, JudgeAssessment

logger = structlog.get_logger()


def get_model() -> Any:
    """Get the LLM model based on configuration.

    Explicitly passes API keys from settings to avoid requiring
    users to export environment variables manually.
    """
    llm_provider = settings.llm_provider

    if llm_provider == "anthropic":
        provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model, provider=provider)

    if llm_provider != "openai":
        logger.warning("Unknown LLM provider, defaulting to OpenAI", provider=llm_provider)

    openai_provider = OpenAIProvider(api_key=settings.openai_api_key)
    return OpenAIModel(settings.openai_model, provider=openai_provider)


class JudgeHandler:
    """
    Handles evidence assessment using an LLM with structured output.

    Uses PydanticAI to ensure responses match the JudgeAssessment schema.
    """

    def __init__(self, model: Any = None) -> None:
        """
        Initialize the JudgeHandler.

        Args:
            model: Optional PydanticAI model. If None, uses config default.
        """
        self.model = model or get_model()
        self.agent = Agent(
            model=self.model,
            output_type=JudgeAssessment,
            system_prompt=SYSTEM_PROMPT,
            retries=3,
        )

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> JudgeAssessment:
        """
        Assess evidence and determine if it's sufficient.

        Args:
            question: The user's research question
            evidence: List of Evidence objects from search

        Returns:
            JudgeAssessment with evaluation results

        Raises:
            JudgeError: If assessment fails after retries
        """
        logger.info(
            "Starting evidence assessment",
            question=question[:100],
            evidence_count=len(evidence),
        )

        # Format the prompt based on whether we have evidence
        if evidence:
            user_prompt = format_user_prompt(question, evidence)
        else:
            user_prompt = format_empty_evidence_prompt(question)

        try:
            # Run the agent with structured output
            result = await self.agent.run(user_prompt)
            assessment = result.output

            logger.info(
                "Assessment complete",
                sufficient=assessment.sufficient,
                recommendation=assessment.recommendation,
                confidence=assessment.confidence,
            )

            return assessment

        except Exception as e:
            logger.error("Assessment failed", error=str(e))
            # Return a safe default assessment on failure
            return self._create_fallback_assessment(question, str(e))

    def _create_fallback_assessment(
        self,
        question: str,
        error: str,
    ) -> JudgeAssessment:
        """
        Create a fallback assessment when LLM fails.

        Args:
            question: The original question
            error: The error message

        Returns:
            Safe fallback JudgeAssessment
        """
        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="Assessment failed due to LLM error",
                clinical_evidence_score=0,
                clinical_reasoning="Assessment failed due to LLM error",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=[
                f"{question} mechanism",
                f"{question} clinical trials",
                f"{question} drug candidates",
            ],
            reasoning=f"Assessment failed: {error}. Recommend retrying with refined queries.",
        )


class MockJudgeHandler:
    """
    Mock JudgeHandler for demo mode without LLM calls.

    Extracts meaningful information from real search results
    to provide a useful demo experience without requiring API keys.
    """

    def __init__(self, mock_response: JudgeAssessment | None = None) -> None:
        """
        Initialize with optional mock response.

        Args:
            mock_response: The assessment to return. If None, extracts from evidence.
        """
        self.mock_response = mock_response
        self.call_count = 0
        self.last_question: str | None = None
        self.last_evidence: list[Evidence] | None = None

    def _extract_key_findings(self, evidence: list[Evidence], max_findings: int = 5) -> list[str]:
        """Extract key findings from evidence titles."""
        findings = []
        for e in evidence[:max_findings]:
            # Use first 150 chars of title as a finding
            title = e.citation.title
            if len(title) > 150:
                title = title[:147] + "..."
            findings.append(title)
        return findings if findings else ["No specific findings extracted (demo mode)"]

    def _extract_drug_candidates(self, question: str, evidence: list[Evidence]) -> list[str]:
        """Extract drug candidates - demo mode returns honest message."""
        # Don't attempt heuristic extraction - it produces garbage like "Oral", "Kidney"
        # Real drug extraction requires LLM analysis
        return [
            "Drug identification requires AI analysis",
            "Enter API key above for full results",
        ]

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> JudgeAssessment:
        """Return assessment based on actual evidence (demo mode)."""
        self.call_count += 1
        self.last_question = question
        self.last_evidence = evidence

        if self.mock_response:
            return self.mock_response

        min_evidence = 3
        evidence_count = len(evidence)

        # Extract meaningful data from actual evidence
        drug_candidates = self._extract_drug_candidates(question, evidence)
        key_findings = self._extract_key_findings(evidence)

        # Calculate scores based on evidence quantity
        mechanism_score = min(10, evidence_count * 2) if evidence_count > 0 else 0
        clinical_score = min(10, evidence_count) if evidence_count > 0 else 0

        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=mechanism_score,
                mechanism_reasoning=(
                    f"Demo mode: Found {evidence_count} sources. "
                    "Configure LLM API key for detailed mechanism analysis."
                ),
                clinical_evidence_score=clinical_score,
                clinical_reasoning=(
                    f"Demo mode: {evidence_count} sources retrieved from PubMed, "
                    "ClinicalTrials.gov, and bioRxiv. Full analysis requires LLM API key."
                ),
                drug_candidates=drug_candidates,
                key_findings=key_findings,
            ),
            sufficient=evidence_count >= min_evidence,
            confidence=min(0.5, evidence_count * 0.1) if evidence_count > 0 else 0.0,
            recommendation="synthesize" if evidence_count >= min_evidence else "continue",
            next_search_queries=(
                [f"{question} mechanism", f"{question} clinical trials"]
                if evidence_count < min_evidence
                else []
            ),
            reasoning=(
                f"Demo mode assessment based on {evidence_count} real search results. "
                "For AI-powered analysis with drug candidate identification and "
                "evidence synthesis, configure OPENAI_API_KEY or ANTHROPIC_API_KEY."
            ),
        )
