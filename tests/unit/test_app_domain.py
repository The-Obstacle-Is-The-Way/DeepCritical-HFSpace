"""Tests for App domain support (SPEC-16: Unified Architecture)."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.app import configure_orchestrator, research_agent
from src.config.domain import ResearchDomain

pytestmark = pytest.mark.unit


class TestAppDomain:
    """Test domain parameter handling in app.py."""

    @patch("src.app.create_orchestrator")
    def test_configure_orchestrator_passes_domain(self, mock_create):
        """Test domain is passed to create_orchestrator (SPEC-16: unified architecture)."""
        # Mock return value
        mock_orch = MagicMock()
        mock_create.return_value = mock_orch

        configure_orchestrator(
            use_mock=False,
            mode="advanced",  # SPEC-16: always advanced
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

        mock_create.assert_called_with(
            config=ANY,
            mode="advanced",
            api_key=None,
            domain=ResearchDomain.SEXUAL_HEALTH,
        )

    @patch("src.app.create_orchestrator")
    def test_configure_orchestrator_with_api_key(self, mock_create):
        """Test API key is passed through."""
        mock_orch = MagicMock()
        mock_create.return_value = mock_orch

        configure_orchestrator(
            use_mock=False,
            user_api_key="sk-test-key",
            domain="sexual_health",
        )

        mock_create.assert_called_with(
            config=ANY,
            mode="advanced",
            api_key="sk-test-key",
            domain="sexual_health",
        )

    @pytest.mark.asyncio
    @patch("src.app.settings")
    @patch("src.app.configure_orchestrator")
    async def test_research_agent_passes_domain(self, mock_config, mock_settings):
        """Test research_agent passes domain to configure_orchestrator."""
        # Mock settings to have some state
        mock_settings.has_openai_key = False
        mock_settings.has_anthropic_key = False

        # Mock orchestrator
        mock_orch = MagicMock()

        # Mock async generator
        async def async_gen(*args):
            if False:
                yield  # Make it a generator

        mock_orch.run = async_gen
        mock_config.return_value = (mock_orch, "Test Backend")

        # SPEC-16: mode parameter removed from research_agent
        gen = research_agent(
            message="query",
            history=[],
            domain=ResearchDomain.SEXUAL_HEALTH.value,
        )

        async for _ in gen:
            pass

        # SPEC-16: mode is always "advanced"
        mock_config.assert_called_with(
            use_mock=False,
            mode="advanced",
            user_api_key=None,
            domain=ResearchDomain.SEXUAL_HEALTH.value,
        )
