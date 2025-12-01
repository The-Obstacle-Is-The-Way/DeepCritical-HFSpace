"""Tests for Magentic Judge termination logic (SPEC-16)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.magentic_agents import create_judge_agent

pytestmark = pytest.mark.unit

# Skip if agent-framework-core not installed
pytest.importorskip("agent_framework")


def test_judge_agent_has_termination_instructions() -> None:
    """Judge agent must be created with explicit instructions for early termination."""
    with patch("src.agents.magentic_agents.get_domain_config") as mock_config:
        # Mock config to return test prompts
        mock_config.return_value.judge_system_prompt = "Test judge prompt"

        with patch("src.agents.magentic_agents.get_chat_client") as mock_client:
            mock_client.return_value = MagicMock()

            with patch("src.agents.magentic_agents.ChatAgent") as mock_chat_agent_cls:
                create_judge_agent()

                # Verify ChatAgent was initialized with correct instructions
                assert mock_chat_agent_cls.called
                call_kwargs = mock_chat_agent_cls.call_args.kwargs
                instructions = call_kwargs.get("instructions", "")

                # Verify critical sections for SPEC-15 termination
                assert "CRITICAL OUTPUT FORMAT" in instructions
                assert "SUFFICIENT EVIDENCE" in instructions
                assert "confidence >= 70%" in instructions
                assert "STOP SEARCHING" in instructions
                assert "Delegate to ReportAgent NOW" in instructions


def test_judge_agent_uses_reasoning_temperature() -> None:
    """Judge agent should be initialized with temperature=1.0 for reasoning models."""
    with patch("src.agents.magentic_agents.get_chat_client") as mock_client:
        mock_client.return_value = MagicMock()

        with patch("src.agents.magentic_agents.ChatAgent") as mock_chat_agent_cls:
            create_judge_agent()

            call_kwargs = mock_chat_agent_cls.call_args.kwargs
            assert call_kwargs.get("temperature") == 1.0


def test_judge_agent_accepts_custom_chat_client() -> None:
    """Judge agent should accept custom chat_client parameter (SPEC-16)."""
    custom_client = MagicMock()

    with patch("src.agents.magentic_agents.ChatAgent") as mock_chat_agent_cls:
        create_judge_agent(chat_client=custom_client)

        call_kwargs = mock_chat_agent_cls.call_args.kwargs
        assert call_kwargs.get("chat_client") == custom_client
