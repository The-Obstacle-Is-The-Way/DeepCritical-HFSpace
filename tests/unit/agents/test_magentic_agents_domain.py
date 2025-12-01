"""Tests for Magentic Agents domain support."""

from unittest.mock import patch

from src.agents.magentic_agents import (
    create_hypothesis_agent,
    create_judge_agent,
    create_report_agent,
    create_search_agent,
)
from src.config.domain import SEXUAL_HEALTH_CONFIG, ResearchDomain


class TestMagenticAgentsDomain:
    @patch("src.agents.magentic_agents.ChatAgent")
    @patch("src.agents.magentic_agents.get_chat_client")
    def test_create_search_agent_uses_domain(self, mock_get_client, mock_agent_cls):
        create_search_agent(domain=ResearchDomain.SEXUAL_HEALTH)

        # Check instructions or description passed to ChatAgent
        call_kwargs = mock_agent_cls.call_args.kwargs
        assert SEXUAL_HEALTH_CONFIG.search_agent_description in call_kwargs["description"]
        # Ideally check instructions too if we update them

    @patch("src.agents.magentic_agents.ChatAgent")
    @patch("src.agents.magentic_agents.get_chat_client")
    def test_create_judge_agent_uses_domain(self, mock_get_client, mock_agent_cls):
        create_judge_agent(domain=ResearchDomain.SEXUAL_HEALTH)

        # Verify domain-specific judge system prompt is passed through
        call_kwargs = mock_agent_cls.call_args.kwargs
        assert SEXUAL_HEALTH_CONFIG.judge_system_prompt in call_kwargs["instructions"]

    @patch("src.agents.magentic_agents.ChatAgent")
    @patch("src.agents.magentic_agents.get_chat_client")
    def test_create_hypothesis_agent_uses_domain(self, mock_get_client, mock_agent_cls):
        create_hypothesis_agent(domain=ResearchDomain.SEXUAL_HEALTH)
        call_kwargs = mock_agent_cls.call_args.kwargs
        assert SEXUAL_HEALTH_CONFIG.hypothesis_agent_description in call_kwargs["description"]

    @patch("src.agents.magentic_agents.ChatAgent")
    @patch("src.agents.magentic_agents.get_chat_client")
    def test_create_report_agent_uses_domain(self, mock_get_client, mock_agent_cls):
        create_report_agent(domain=ResearchDomain.SEXUAL_HEALTH)
        # Check instructions contains domain prompt
        call_kwargs = mock_agent_cls.call_args.kwargs
        assert SEXUAL_HEALTH_CONFIG.report_system_prompt in call_kwargs["instructions"]
