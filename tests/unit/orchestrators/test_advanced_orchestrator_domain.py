"""Tests for Advanced Orchestrator domain support."""

from unittest.mock import MagicMock, patch

from src.config.domain import ResearchDomain
from src.orchestrators.advanced import AdvancedOrchestrator


class TestAdvancedOrchestratorDomain:
    @patch("src.orchestrators.advanced.get_chat_client")
    def test_advanced_orchestrator_accepts_domain(self, mock_get_client):
        # Mock to avoid API key validation
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        orch = AdvancedOrchestrator(domain=ResearchDomain.SEXUAL_HEALTH, api_key="sk-test")
        assert orch.domain == ResearchDomain.SEXUAL_HEALTH

    @patch("src.orchestrators.advanced.create_search_agent")
    @patch("src.orchestrators.advanced.create_judge_agent")
    @patch("src.orchestrators.advanced.create_hypothesis_agent")
    @patch("src.orchestrators.advanced.create_report_agent")
    @patch("src.orchestrators.advanced.MagenticBuilder")
    @patch("src.orchestrators.advanced.get_chat_client")
    def test_build_workflow_uses_domain(
        self,
        mock_get_client,
        mock_builder,
        mock_create_report,
        mock_create_hypothesis,
        mock_create_judge,
        mock_create_search,
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        orch = AdvancedOrchestrator(domain=ResearchDomain.SEXUAL_HEALTH, api_key="sk-test")

        # Call private method to verify agent creation calls
        orch._build_workflow()

        # Verify agents created with domain and correct client
        mock_create_search.assert_called_with(mock_client, domain=ResearchDomain.SEXUAL_HEALTH)
        mock_create_judge.assert_called_with(mock_client, domain=ResearchDomain.SEXUAL_HEALTH)
        mock_create_hypothesis.assert_called_with(mock_client, domain=ResearchDomain.SEXUAL_HEALTH)
        mock_create_report.assert_called_with(mock_client, domain=ResearchDomain.SEXUAL_HEALTH)
