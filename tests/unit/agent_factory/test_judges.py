"""Unit tests for JudgeHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.utils.models import AssessmentDetails, Citation, Evidence, JudgeAssessment


class TestJudgeHandler:
    """Tests for JudgeHandler."""

    @pytest.mark.asyncio
    async def test_assess_returns_assessment(self):
        """JudgeHandler should return JudgeAssessment from LLM."""
        # Create mock assessment
        expected_confidence = 0.85
        mock_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=8,
                mechanism_reasoning="Strong mechanistic evidence",
                clinical_evidence_score=7,
                clinical_reasoning="Good clinical support",
                drug_candidates=["Metformin"],
                key_findings=["Neuroprotective effects"],
            ),
            sufficient=True,
            confidence=expected_confidence,
            recommendation="synthesize",
            next_search_queries=[],
            reasoning="Evidence is sufficient for synthesis",
        )

        # Mock the PydanticAI agent
        mock_result = MagicMock()
        mock_result.output = mock_assessment

        with (
            patch("src.agent_factory.judges.get_model") as mock_get_model,
            patch("src.agent_factory.judges.Agent") as mock_agent_class,
        ):
            mock_get_model.return_value = MagicMock()
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            # Replace the agent with our mock
            handler.agent = mock_agent

            evidence = [
                Evidence(
                    content="Metformin shows neuroprotective properties...",
                    citation=Citation(
                        source="pubmed",
                        title="Metformin in AD",
                        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                        date="2024-01-01",
                    ),
                )
            ]

            result = await handler.assess("metformin alzheimer", evidence)

            assert result.sufficient is True
            assert result.recommendation == "synthesize"
            assert result.confidence == expected_confidence
            assert "Metformin" in result.details.drug_candidates

    @pytest.mark.asyncio
    async def test_assess_empty_evidence(self):
        """JudgeHandler should handle empty evidence gracefully."""
        mock_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="No evidence to assess",
                clinical_evidence_score=0,
                clinical_reasoning="No evidence to assess",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=["metformin alzheimer mechanism"],
            reasoning="No evidence found, need to search more",
        )

        mock_result = MagicMock()
        mock_result.output = mock_assessment

        with (
            patch("src.agent_factory.judges.get_model") as mock_get_model,
            patch("src.agent_factory.judges.Agent") as mock_agent_class,
        ):
            mock_get_model.return_value = MagicMock()
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            handler.agent = mock_agent

            result = await handler.assess("metformin alzheimer", [])

            assert result.sufficient is False
            assert result.recommendation == "continue"
            assert len(result.next_search_queries) > 0

    @pytest.mark.asyncio
    async def test_assess_handles_llm_failure(self):
        """JudgeHandler should return fallback on LLM failure."""
        with (
            patch("src.agent_factory.judges.get_model") as mock_get_model,
            patch("src.agent_factory.judges.Agent") as mock_agent_class,
        ):
            mock_get_model.return_value = MagicMock()
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(side_effect=Exception("API Error"))
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            handler.agent = mock_agent

            evidence = [
                Evidence(
                    content="Some content",
                    citation=Citation(
                        source="pubmed",
                        title="Title",
                        url="url",
                        date="2024",
                    ),
                )
            ]

            result = await handler.assess("test question", evidence)

            # Should return fallback, not raise
            assert result.sufficient is False
            assert result.recommendation == "continue"
            assert "failed" in result.reasoning.lower()


class TestMockJudgeHandler:
    """Tests for MockJudgeHandler."""

    @pytest.mark.asyncio
    async def test_mock_handler_returns_default(self):
        """MockJudgeHandler should return default assessment."""
        handler = MockJudgeHandler()

        evidence = [
            Evidence(
                content="Content 1",
                citation=Citation(source="pubmed", title="T1", url="u1", date="2024"),
            ),
            Evidence(
                content="Content 2",
                citation=Citation(source="pubmed", title="T2", url="u2", date="2024"),
            ),
        ]

        result = await handler.assess("test", evidence)

        expected_mech_score = 7
        expected_evidence_len = 2

        assert handler.call_count == 1
        assert handler.last_question == "test"
        assert handler.last_evidence is not None
        assert len(handler.last_evidence) == expected_evidence_len
        assert result.details.mechanism_score == expected_mech_score
        assert result.sufficient is False
        assert result.recommendation == "continue"

    @pytest.mark.asyncio
    async def test_mock_handler_custom_response(self):
        """MockJudgeHandler should return custom response when provided."""
        expected_score = 10
        custom_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=expected_score,
                mechanism_reasoning="Custom reasoning",
                clinical_evidence_score=expected_score,
                clinical_reasoning="Custom clinical",
                drug_candidates=["CustomDrug"],
                key_findings=["Custom finding"],
            ),
            sufficient=True,
            confidence=1.0,
            recommendation="synthesize",
            next_search_queries=[],
            reasoning="Custom assessment logic for testing purposes must be at least 20 chars long",
        )

        handler = MockJudgeHandler(mock_response=custom_assessment)
        result = await handler.assess("test", [])

        assert result.details.mechanism_score == expected_score
        assert result.details.drug_candidates == ["CustomDrug"]

    @pytest.mark.asyncio
    async def test_mock_handler_insufficient_with_few_evidence(self):
        """MockJudgeHandler should recommend continue with < 3 evidence."""
        handler = MockJudgeHandler()

        # Only 2 pieces of evidence
        evidence = [
            Evidence(
                content="Content",
                citation=Citation(source="pubmed", title="T", url="u", date="2024"),
            ),
            Evidence(
                content="Content 2",
                citation=Citation(source="pubmed", title="T2", url="u2", date="2024"),
            ),
        ]

        result = await handler.assess("test", evidence)

        assert result.sufficient is False
        assert result.recommendation == "continue"
        assert len(result.next_search_queries) > 0
