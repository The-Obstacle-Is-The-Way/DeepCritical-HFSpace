"""Tests for judge prompt domain support."""

from src.config.domain import SEXUAL_HEALTH_CONFIG, ResearchDomain
from src.prompts.judge import format_user_prompt, get_scoring_prompt, get_system_prompt


class TestJudgePromptDomain:
    def test_get_system_prompt_default(self):
        prompt = get_system_prompt()
        assert SEXUAL_HEALTH_CONFIG.name in prompt
        assert "You are an expert research judge" in prompt

    def test_get_system_prompt_sexual_health(self):
        prompt = get_system_prompt(ResearchDomain.SEXUAL_HEALTH)
        assert SEXUAL_HEALTH_CONFIG.name in prompt
        assert "sexual health" in prompt.lower()
        assert "You are an expert research judge" in prompt

    def test_get_scoring_prompt_default(self):
        prompt = get_scoring_prompt()
        assert "Score this evidence for relevance" in prompt

    def test_format_user_prompt_default(self):
        prompt = format_user_prompt("query", [])
        assert "Score this evidence for relevance" in prompt

    def test_format_user_prompt_with_domain(self):
        prompt = format_user_prompt("query", [], domain=ResearchDomain.SEXUAL_HEALTH)
        assert "Score this evidence for relevance" in prompt
