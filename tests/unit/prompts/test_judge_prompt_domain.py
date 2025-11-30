"""Tests for judge prompt domain support."""

from src.config.domain import DRUG_REPURPOSING_CONFIG, GENERAL_CONFIG, ResearchDomain
from src.prompts.judge import format_user_prompt, get_scoring_prompt, get_system_prompt


class TestJudgePromptDomain:
    def test_get_system_prompt_default(self):
        prompt = get_system_prompt()
        assert GENERAL_CONFIG.judge_system_prompt in prompt
        assert "Your task is to SCORE evidence" in prompt

    def test_get_system_prompt_domain(self):
        prompt = get_system_prompt(ResearchDomain.DRUG_REPURPOSING)
        assert DRUG_REPURPOSING_CONFIG.judge_system_prompt in prompt
        assert "Your task is to SCORE evidence" in prompt

    def test_get_scoring_prompt_default(self):
        prompt = get_scoring_prompt()
        assert GENERAL_CONFIG.judge_scoring_prompt == prompt

    def test_format_user_prompt_default(self):
        prompt = format_user_prompt("query", [])
        assert GENERAL_CONFIG.judge_scoring_prompt in prompt
        assert "drug repurposing" not in prompt.lower()

    def test_format_user_prompt_with_domain(self):
        prompt = format_user_prompt("query", [], domain=ResearchDomain.DRUG_REPURPOSING)
        assert DRUG_REPURPOSING_CONFIG.judge_scoring_prompt in prompt
        # The drug repurposing prompt contains "drug repurposing"
        assert "drug repurposing" in prompt.lower()
