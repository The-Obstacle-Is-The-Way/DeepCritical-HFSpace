"""Centralized domain configuration for research agents.

This module defines research domains and their associated prompts,
allowing the agent to operate in domain-agnostic or domain-specific modes.

Usage:
    from src.config.domain import get_domain_config, ResearchDomain

    # Get default config
    config = get_domain_config()

    # Get specific domain
    config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)

    # Use in prompts
    system_prompt = config.judge_system_prompt
"""

from enum import Enum

from pydantic import BaseModel


class ResearchDomain(str, Enum):
    """Available research domains.

    DeepBoner is a focused Sexual Health Research Specialist.
    Only SEXUAL_HEALTH domain is supported.
    """

    SEXUAL_HEALTH = "sexual_health"


class DomainConfig(BaseModel):
    """Configuration for a research domain.

    Contains all domain-specific text used across the codebase,
    ensuring consistency and single-source-of-truth.
    """

    # Identity
    name: str
    description: str

    # Report generation
    report_title: str
    report_focus: str

    # Judge prompts
    judge_system_prompt: str
    judge_scoring_prompt: str

    # Hypothesis prompts
    hypothesis_system_prompt: str

    # Report writer prompts
    report_system_prompt: str

    # Search context
    search_description: str
    search_example_query: str

    # Agent descriptions (for Magentic mode)
    search_agent_description: str
    hypothesis_agent_description: str


# ─────────────────────────────────────────────────────────────────
# Domain Configuration - Sexual Health Only
# ─────────────────────────────────────────────────────────────────

SEXUAL_HEALTH_CONFIG = DomainConfig(
    name="Sexual Health Research",
    description="Sexual health and wellness research specialist",
    report_title="## Sexual Health Analysis",
    report_focus="sexual health and wellness interventions",
    judge_system_prompt="""You are an expert sexual health research judge.
Your role is to evaluate evidence for sexual health interventions, assess
efficacy and safety data, and determine clinical applicability.""",
    judge_scoring_prompt="""Score this evidence for sexual health relevance.
Provide ONLY scores and extracted data.""",
    hypothesis_system_prompt=(
        """You are a biomedical research scientist specializing in sexual health.
Your role is to generate evidence-based hypotheses for sexual health interventions,
identifying mechanisms of action and potential therapeutic applications."""
    ),
    report_system_prompt=(
        """You are a scientific writer specializing in sexual health research reports.
Your role is to synthesize evidence into clear recommendations for sexual health
interventions with proper safety considerations."""
    ),
    search_description="Searches biomedical literature for sexual health evidence",
    search_example_query="testosterone therapy female libido",
    search_agent_description="Searches PubMed for sexual health evidence",
    hypothesis_agent_description="Generates hypotheses for sexual health interventions",
)

# ─────────────────────────────────────────────────────────────────
# Domain Registry
# ─────────────────────────────────────────────────────────────────

DOMAIN_CONFIGS: dict[ResearchDomain, DomainConfig] = {
    ResearchDomain.SEXUAL_HEALTH: SEXUAL_HEALTH_CONFIG,
}

# Default domain - DeepBoner is Sexual Health focused
DEFAULT_DOMAIN = ResearchDomain.SEXUAL_HEALTH


def get_domain_config(domain: ResearchDomain | str | None = None) -> DomainConfig:
    """Get configuration for a research domain.

    Args:
        domain: The research domain. Defaults to sexual_health if None.

    Returns:
        DomainConfig for the specified domain.
    """
    if domain is None:
        domain = DEFAULT_DOMAIN

    if isinstance(domain, str):
        try:
            domain = ResearchDomain(domain)
        except ValueError as e:
            valid_domains = [d.value for d in ResearchDomain]
            raise ValueError(f"Invalid domain '{domain}'. Valid domains: {valid_domains}") from e

    return DOMAIN_CONFIGS[domain]
