"""Tests for report prompt domain support."""

from src.config.domain import SEXUAL_HEALTH_CONFIG, ResearchDomain
from src.prompts.report import get_system_prompt


class TestReportPromptDomain:
    def test_get_system_prompt_default(self):
        prompt = get_system_prompt()
        assert SEXUAL_HEALTH_CONFIG.name in prompt
        assert "Your role is to synthesize evidence" in prompt

    def test_get_system_prompt_sexual_health(self):
        prompt = get_system_prompt(ResearchDomain.SEXUAL_HEALTH)
        assert SEXUAL_HEALTH_CONFIG.name in prompt
        assert "sexual health" in prompt.lower()
        assert "Your role is to synthesize evidence" in prompt
