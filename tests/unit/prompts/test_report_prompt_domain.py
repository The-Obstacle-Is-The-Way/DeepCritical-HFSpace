"""Tests for report prompt domain support."""

from src.config.domain import DRUG_REPURPOSING_CONFIG, GENERAL_CONFIG, ResearchDomain
from src.prompts.report import get_system_prompt


class TestReportPromptDomain:
    def test_get_system_prompt_default(self):
        prompt = get_system_prompt()
        assert GENERAL_CONFIG.report_system_prompt in prompt
        assert "Your role is to synthesize evidence" in prompt

    def test_get_system_prompt_domain(self):
        prompt = get_system_prompt(ResearchDomain.DRUG_REPURPOSING)
        assert DRUG_REPURPOSING_CONFIG.report_system_prompt in prompt
        assert "Your role is to synthesize evidence" in prompt
