"""Centralized domain configuration for research agents.

This module defines research domains and their associated prompts,
allowing the agent to operate in domain-agnostic or domain-specific modes.

Usage:
    from src.config.domain import get_domain_config, ResearchDomain

    # Get default (general) config
    config = get_domain_config()

    # Get specific domain
    config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)

    # Use in prompts
    system_prompt = config.judge_system_prompt
"""

from enum import Enum

from pydantic import BaseModel


class ResearchDomain(str, Enum):
    """Available research domains."""

    GENERAL = "general"
    DRUG_REPURPOSING = "drug_repurposing"
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
# Domain Definitions
# ─────────────────────────────────────────────────────────────────

GENERAL_CONFIG = DomainConfig(
    name="General Research",
    description="General-purpose biomedical research agent",
    report_title="## Research Analysis",
    report_focus="comprehensive research synthesis",
    judge_system_prompt="""You are an expert research judge.
Your role is to evaluate evidence quality, assess relevance to the research query,
and determine if sufficient evidence exists to synthesize findings.""",
    judge_scoring_prompt="""Score this evidence for research relevance.
Provide ONLY scores and extracted data.""",
    hypothesis_system_prompt="""You are a biomedical research scientist.
Your role is to generate evidence-based hypotheses from the literature,
identifying key mechanisms, targets, and potential therapeutic implications.""",
    report_system_prompt="""You are a scientific writer specializing in research reports.
Your role is to synthesize evidence into clear, well-structured reports with
proper citations and evidence-based conclusions.""",
    search_description="Searches biomedical literature for relevant evidence",
    search_example_query="metformin aging mechanisms",
    search_agent_description="Searches PubMed, ClinicalTrials.gov, and Europe PMC for evidence",
    hypothesis_agent_description="Generates mechanistic hypotheses from evidence",
)

DRUG_REPURPOSING_CONFIG = DomainConfig(
    name="Drug Repurposing",
    description="Drug repurposing research specialist",
    report_title="## Drug Repurposing Analysis",
    report_focus="drug repurposing opportunities",
    judge_system_prompt="""You are an expert drug repurposing research judge.
Your role is to evaluate evidence for drug repurposing potential, assess
mechanism plausibility, and determine if compounds warrant further investigation.""",
    judge_scoring_prompt="""Score this evidence for drug repurposing potential.
Provide ONLY scores and extracted data.""",
    hypothesis_system_prompt=(
        """You are a biomedical research scientist specializing in drug repurposing.
Your role is to generate mechanistic hypotheses for how existing drugs might
treat new indications, based on shared pathways and targets."""
    ),
    report_system_prompt=(
        """You are a scientific writer specializing in drug repurposing research reports.
Your role is to synthesize evidence into actionable drug repurposing recommendations
with clear mechanistic rationale and clinical translation potential."""
    ),
    search_description="Searches biomedical literature for drug repurposing evidence",
    search_example_query="metformin alzheimer repurposing",
    search_agent_description="Searches PubMed for drug repurposing evidence",
    hypothesis_agent_description="Generates mechanistic hypotheses for drug repurposing",
)

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
    ResearchDomain.GENERAL: GENERAL_CONFIG,
    ResearchDomain.DRUG_REPURPOSING: DRUG_REPURPOSING_CONFIG,
    ResearchDomain.SEXUAL_HEALTH: SEXUAL_HEALTH_CONFIG,
}

# Default domain
DEFAULT_DOMAIN = ResearchDomain.GENERAL


def get_domain_config(domain: ResearchDomain | str | None = None) -> DomainConfig:
    """Get configuration for a research domain.

    Args:
        domain: The research domain. Defaults to GENERAL if None.

    Returns:
        DomainConfig for the specified domain.
    """
    if domain is None:
        domain = DEFAULT_DOMAIN

    if isinstance(domain, str):
        try:
            domain = ResearchDomain(domain)
        except ValueError:
            domain = DEFAULT_DOMAIN

    return DOMAIN_CONFIGS[domain]
