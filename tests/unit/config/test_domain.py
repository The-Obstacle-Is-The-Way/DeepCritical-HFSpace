"""Tests for domain configuration."""

from src.config.domain import (
    SEXUAL_HEALTH_CONFIG,
    ResearchDomain,
    get_domain_config,
)


class TestResearchDomain:
    def test_enum_values(self):
        # DeepBoner only supports sexual health
        assert ResearchDomain.SEXUAL_HEALTH.value == "sexual_health"
        assert len(ResearchDomain) == 1


class TestGetDomainConfig:
    def test_default_returns_sexual_health(self):
        config = get_domain_config()
        assert config.name == "Sexual Health Research"

    def test_explicit_sexual_health(self):
        config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)
        assert "Sexual Health" in config.report_title

    def test_accepts_string(self):
        config = get_domain_config("sexual_health")
        assert "Sexual Health" in config.name

    def test_invalid_string_raises_value_error(self):
        # Invalid domains should fail fast with clear error
        import pytest

        with pytest.raises(ValueError) as exc_info:
            get_domain_config("invalid_domain")
        assert "Invalid domain" in str(exc_info.value)
        assert "sexual_health" in str(exc_info.value)  # Shows valid options

    def test_config_has_required_fields(self):
        required_fields = [
            "name",
            "report_title",
            "search_description",
        ]
        config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)
        for field in required_fields:
            assert getattr(config, field), f"SEXUAL_HEALTH missing {field}"

    def test_sexual_health_config_exported(self):
        # Verify the config constant is exported
        assert SEXUAL_HEALTH_CONFIG.name == "Sexual Health Research"
