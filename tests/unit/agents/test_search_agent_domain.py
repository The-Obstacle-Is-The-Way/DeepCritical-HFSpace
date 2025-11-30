"""Tests for Search Agent domain support."""

from unittest.mock import MagicMock

from src.agents.search_agent import SearchAgent
from src.config.domain import SEXUAL_HEALTH_CONFIG, ResearchDomain


class TestSearchAgentDomain:
    def test_search_agent_accepts_domain(self):
        mock_handler = MagicMock()
        store = {"current": []}

        agent = SearchAgent(
            search_handler=mock_handler, evidence_store=store, domain=ResearchDomain.SEXUAL_HEALTH
        )

        # Verify description updated
        assert agent.description == SEXUAL_HEALTH_CONFIG.search_agent_description
