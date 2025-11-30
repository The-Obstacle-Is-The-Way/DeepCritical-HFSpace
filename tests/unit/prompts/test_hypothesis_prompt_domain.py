"""Tests for hypothesis prompt domain support."""

from src.config.domain import DRUG_REPURPOSING_CONFIG, GENERAL_CONFIG, ResearchDomain
from src.prompts.hypothesis import get_system_prompt


class TestHypothesisPromptDomain:
    def test_get_system_prompt_default(self):
        prompt = get_system_prompt()
        assert GENERAL_CONFIG.hypothesis_system_prompt in prompt
        assert "Your role is to generate mechanistic hypotheses" in prompt

    def test_get_system_prompt_domain(self):
        prompt = get_system_prompt(ResearchDomain.DRUG_REPURPOSING)
        assert DRUG_REPURPOSING_CONFIG.hypothesis_system_prompt in prompt
        assert "Your role is to generate mechanistic hypotheses" in prompt
