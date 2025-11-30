"""Tests for Orchestrator (Simple) domain support."""

from unittest.mock import MagicMock

from src.config.domain import SEXUAL_HEALTH_CONFIG, ResearchDomain
from src.orchestrators.simple import Orchestrator


class TestSimpleOrchestratorDomain:
    def test_orchestrator_accepts_domain(self):
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orch = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

        assert orch.domain == ResearchDomain.SEXUAL_HEALTH
        assert orch.domain_config.name == SEXUAL_HEALTH_CONFIG.name

    def test_orchestrator_uses_domain_title_in_synthesis(self):
        mock_search = MagicMock()
        mock_judge = MagicMock()

        orch = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

        # Test _generate_synthesis
        mock_assessment = MagicMock()
        mock_assessment.details.drug_candidates = []
        mock_assessment.details.key_findings = []
        mock_assessment.confidence = 0.5
        mock_assessment.reasoning = "test"
        mock_assessment.details.mechanism_score = 5
        mock_assessment.details.clinical_evidence_score = 5

        report = orch._generate_synthesis("query", [], mock_assessment)
        assert "## Sexual Health Analysis" in report

        # Test _generate_partial_synthesis
        report_partial = orch._generate_partial_synthesis("query", [])
        assert "## Sexual Health Analysis" in report_partial
