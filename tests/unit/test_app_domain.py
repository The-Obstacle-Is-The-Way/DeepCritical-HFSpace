"""Tests for App domain support."""

from unittest.mock import ANY, MagicMock, patch

from src.app import configure_orchestrator, research_agent
from src.config.domain import ResearchDomain


class TestAppDomain:
    @patch("src.app.create_orchestrator")
    @patch("src.app.MockJudgeHandler")
    def test_configure_orchestrator_passes_domain_mock_mode(self, mock_judge, mock_create):
        """Test domain is passed when using mock mode (unit test path)."""
        configure_orchestrator(use_mock=True, mode="simple", domain=ResearchDomain.SEXUAL_HEALTH)

        # MockJudgeHandler should receive domain
        mock_judge.assert_called_with(domain=ResearchDomain.SEXUAL_HEALTH)
        mock_create.assert_called_with(
            search_handler=ANY,
            judge_handler=ANY,
            config=ANY,
            mode="simple",
            api_key=None,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.app.create_orchestrator")
    @patch("src.app.HFInferenceJudgeHandler")
    def test_configure_orchestrator_passes_domain_free_tier(self, mock_hf_judge, mock_create):
        """Test domain is passed when using free tier (no API keys)."""
        configure_orchestrator(use_mock=False, mode="simple", domain=ResearchDomain.SEXUAL_HEALTH)

        # HFInferenceJudgeHandler should receive domain (no API keys = free tier)
        mock_hf_judge.assert_called_with(domain=ResearchDomain.SEXUAL_HEALTH)
        mock_create.assert_called_with(
            search_handler=ANY,
            judge_handler=ANY,
            config=ANY,
            mode="simple",
            api_key=None,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

    @patch("src.app.configure_orchestrator")
    async def test_research_agent_passes_domain(self, mock_config):
        # Mock orchestrator
        mock_orch = MagicMock()
        mock_orch.run.return_value = []  # Async iterator?

        # To mock async generator
        async def async_gen(*args):
            if False:
                yield  # Make it a generator

        mock_orch.run = async_gen

        mock_config.return_value = (mock_orch, "Test Backend")

        # Consume the generator from research_agent
        gen = research_agent(
            message="query", history=[], mode="simple", domain=ResearchDomain.SEXUAL_HEALTH
        )

        async for _ in gen:
            pass

        mock_config.assert_called_with(
            use_mock=False, mode="simple", user_api_key=None, domain=ResearchDomain.SEXUAL_HEALTH
        )
