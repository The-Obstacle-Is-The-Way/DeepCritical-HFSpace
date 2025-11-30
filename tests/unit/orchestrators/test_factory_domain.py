"""Tests for Orchestrator Factory domain support."""

from unittest.mock import ANY, MagicMock, patch

from src.config.domain import ResearchDomain
from src.orchestrators.factory import create_orchestrator


class TestFactoryDomain:
    @patch("src.orchestrators.factory.Orchestrator")
    def test_create_simple_uses_domain(self, mock_simple_cls):
        mock_search = MagicMock()
        mock_judge = MagicMock()

        create_orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            mode="simple",
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

        mock_simple_cls.assert_called_with(
            search_handler=mock_search,
            judge_handler=mock_judge,
            config=ANY,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

    @patch("src.orchestrators.factory._get_advanced_orchestrator_class")
    def test_create_advanced_uses_domain(self, mock_get_cls):
        mock_adv_cls = MagicMock()
        mock_get_cls.return_value = mock_adv_cls

        create_orchestrator(mode="advanced", domain=ResearchDomain.SEXUAL_HEALTH)

        call_kwargs = mock_adv_cls.call_args.kwargs
        assert call_kwargs["domain"] == ResearchDomain.SEXUAL_HEALTH
