"""Judge handler for evidence assessment using PydanticAI."""

from typing import Any

import structlog
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

from src.prompts.judge import (
    SYSTEM_PROMPT,
    format_empty_evidence_prompt,
    format_user_prompt,
)
from src.utils.config import settings
from src.utils.models import AssessmentDetails, Evidence, JudgeAssessment

logger = structlog.get_logger()


def get_model() -> Any:
    """Get the LLM model based on configuration."""
    provider = settings.llm_provider

    if provider == "anthropic":
        return AnthropicModel(settings.anthropic_model)

    if provider != "openai":
        logger.warning("Unknown LLM provider, defaulting to OpenAI", provider=provider)

    return OpenAIModel(settings.openai_model)


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
    Mock JudgeHandler for testing without LLM calls.

    Use this in unit tests to avoid API calls.
    """

    def __init__(self, mock_response: JudgeAssessment | None = None) -> None:
        """
        Initialize with optional mock response.

        Args:
            mock_response: The assessment to return. If None, uses default.
        """
        self.mock_response = mock_response
        self.call_count = 0
        self.last_question: str | None = None
        self.last_evidence: list[Evidence] | None = None

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> JudgeAssessment:
        """Return the mock response."""
        self.call_count += 1
        self.last_question = question
        self.last_evidence = evidence

        if self.mock_response:
            return self.mock_response

        min_evidence = 3
        # Default mock response
        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=7,
                mechanism_reasoning="Mock assessment - good mechanism evidence",
                clinical_evidence_score=6,
                clinical_reasoning="Mock assessment - moderate clinical evidence",
                drug_candidates=["Drug A", "Drug B"],
                key_findings=["Finding 1", "Finding 2"],
            ),
            sufficient=len(evidence) >= min_evidence,
            confidence=0.75,
            recommendation="synthesize" if len(evidence) >= min_evidence else "continue",
            next_search_queries=["query 1", "query 2"] if len(evidence) < min_evidence else [],
            reasoning="Mock assessment for testing purposes",
        )
