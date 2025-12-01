"""Tests for Magentic Judge termination logic."""

from unittest.mock import patch

from src.agents.magentic_agents import create_judge_agent


def test_judge_agent_has_termination_instructions() -> None:
    """Judge agent must be created with explicit instructions for early termination."""
    with patch("src.agents.magentic_agents.get_domain_config") as mock_config:
        # Mock config to return empty strings so we test the hardcoded critical section
        mock_config.return_value.judge_system_prompt = ""

        with patch("src.agents.magentic_agents.ChatAgent") as mock_chat_agent_cls:
            with patch("src.agents.magentic_agents.settings") as mock_settings:
                mock_settings.openai_api_key = "sk-dummy"
                mock_settings.openai_model = "gpt-4"

                create_judge_agent()

                # Verify ChatAgent was initialized with correct instructions
                assert mock_chat_agent_cls.called
                call_kwargs = mock_chat_agent_cls.call_args.kwargs
                instructions = call_kwargs.get("instructions", "")

                # Verify critical sections from Solution B
                assert "CRITICAL OUTPUT FORMAT" in instructions
                assert "SUFFICIENT EVIDENCE" in instructions
                assert "confidence >= 70%" in instructions
                assert "STOP SEARCHING" in instructions
                assert "Delegate to ReportAgent NOW" in instructions


def test_judge_agent_uses_reasoning_temperature() -> None:
    """Judge agent should be initialized with temperature=1.0."""
    with patch("src.agents.magentic_agents.ChatAgent") as mock_chat_agent_cls:
        with patch("src.agents.magentic_agents.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-dummy"
            mock_settings.openai_model = "gpt-4"

            create_judge_agent()

            call_kwargs = mock_chat_agent_cls.call_args.kwargs
            assert call_kwargs.get("temperature") == 1.0
