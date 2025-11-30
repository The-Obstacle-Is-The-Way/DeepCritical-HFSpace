"""Tests for JudgeHandler domain support."""

from unittest.mock import MagicMock, patch

from src.agent_factory.judges import JudgeHandler
from src.config.domain import ResearchDomain
from src.utils.models import AssessmentDetails, JudgeAssessment


class TestJudgeHandlerDomain:
    @patch("src.agent_factory.judges.get_model")
    @patch("src.agent_factory.judges.Agent")
    def test_judge_handler_accepts_domain(self, mock_agent_cls, mock_get_model):
        # Mock get_model to avoid API key requirement
        mock_get_model.return_value = MagicMock()
        # Test init with domain
        handler = JudgeHandler(domain=ResearchDomain.SEXUAL_HEALTH)
        assert handler.domain == ResearchDomain.SEXUAL_HEALTH

    @patch("src.agent_factory.judges.get_model")
    @patch("src.agent_factory.judges.Agent")
    @patch("src.agent_factory.judges.format_user_prompt")
    @patch("src.agent_factory.judges.select_evidence_for_judge")
    async def test_judge_handler_passes_domain_to_prompt(
        self, mock_select, mock_format, mock_agent_cls, mock_get_model
    ):
        # Setup mocks
        mock_get_model.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        mock_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="Insufficient evidence to determine mechanism.",
                clinical_evidence_score=0,
                clinical_reasoning="Insufficient evidence to determine clinical viability.",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=[],
            reasoning=("Insufficient evidence collected so far to form a conclusion."),
        )

        # Use async return value for run()
        async def mock_run(*args, **kwargs):
            return MagicMock(output=mock_assessment)

        mock_agent_instance.run.side_effect = mock_run

        mock_select.return_value = []  # mock select returns empty list
        # Wait, if evidence is empty, format_empty_evidence_prompt is called.
        # We want format_user_prompt to be called.

        evidence = [MagicMock()]  # Provide some evidence
        mock_select.return_value = evidence

        # Test
        handler = JudgeHandler(domain=ResearchDomain.DRUG_REPURPOSING)
        await handler.assess("query", evidence)

        # Verify format_user_prompt called with domain
        mock_format.assert_called_once()
        call_kwargs = mock_format.call_args.kwargs
        # Or check args if positional
        # format_user_prompt signature: (question, evidence, iteration, max_iterations, ...)

        # Check if domain was passed in kwargs
        assert call_kwargs.get("domain") == ResearchDomain.DRUG_REPURPOSING
