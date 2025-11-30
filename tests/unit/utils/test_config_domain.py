"""Tests for research domain configuration settings."""

from src.config.domain import ResearchDomain
from src.utils.config import Settings


def test_research_domain_default():
    settings = Settings()
    assert settings.research_domain == ResearchDomain.GENERAL


def test_research_domain_from_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_DOMAIN", "drug_repurposing")
    settings = Settings()
    assert settings.research_domain == ResearchDomain.DRUG_REPURPOSING
