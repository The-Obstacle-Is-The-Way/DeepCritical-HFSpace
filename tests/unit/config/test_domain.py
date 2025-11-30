"""Tests for domain configuration."""

from src.config.domain import (
    ResearchDomain,
    get_domain_config,
)


class TestResearchDomain:
    def test_enum_values(self):
        assert ResearchDomain.GENERAL.value == "general"
        assert ResearchDomain.DRUG_REPURPOSING.value == "drug_repurposing"
        assert ResearchDomain.SEXUAL_HEALTH.value == "sexual_health"


class TestGetDomainConfig:
    def test_default_returns_general(self):
        config = get_domain_config()
        assert config.name == "General Research"

    def test_explicit_general(self):
        config = get_domain_config(ResearchDomain.GENERAL)
        assert "Research Analysis" in config.report_title

    def test_drug_repurposing(self):
        config = get_domain_config(ResearchDomain.DRUG_REPURPOSING)
        assert "Drug Repurposing" in config.report_title
        assert "drug repurposing" in config.judge_system_prompt.lower()

    def test_sexual_health(self):
        config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)
        assert "Sexual Health" in config.report_title

    def test_accepts_string(self):
        config = get_domain_config("drug_repurposing")
        assert "Drug Repurposing" in config.name

    def test_invalid_string_returns_default(self):
        config = get_domain_config("invalid_domain")
        assert config.name == "General Research"

    def test_all_domains_have_required_fields(self):
        required_fields = [
            "name",
            "report_title",
            "judge_system_prompt",
            "hypothesis_system_prompt",
            "report_system_prompt",
        ]
        for domain in ResearchDomain:
            config = get_domain_config(domain)
            for field in required_fields:
                assert getattr(config, field), f"{domain} missing {field}"
