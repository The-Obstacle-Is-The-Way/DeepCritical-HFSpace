"""Magentic-compatible agents using ChatAgent pattern."""

from agent_framework import ChatAgent

from src.agents.tools import (
    get_bibliography,
    search_clinical_trials,
    search_preprints,
    search_pubmed,
)
from src.clients.base import BaseChatClient
from src.clients.factory import get_chat_client
from src.config.domain import ResearchDomain, get_domain_config
from src.prompts.hypothesis import get_system_prompt as get_hypothesis_prompt
from src.prompts.judge import get_system_prompt as get_judge_prompt
from src.prompts.report import get_system_prompt as get_report_prompt
from src.prompts.search import get_system_prompt as get_search_prompt


def create_search_agent(
    chat_client: BaseChatClient | None = None,
    domain: ResearchDomain | str | None = None,
    api_key: str | None = None,
) -> ChatAgent:
    """Create a search agent with internal LLM and search tools.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.
        api_key: Optional BYOK key (auto-detects provider from prefix).

    Returns:
        ChatAgent configured for biomedical search
    """
    client = chat_client or get_chat_client(api_key=api_key)
    config = get_domain_config(domain)

    return ChatAgent(
        name="SearchAgent",
        description=config.search_agent_description,
        instructions=get_search_prompt(domain),
        chat_client=client,
        tools=[search_pubmed, search_clinical_trials, search_preprints],
        temperature=1.0,  # Explicitly set for reasoning model compatibility (o1/o3)
    )


def create_judge_agent(
    chat_client: BaseChatClient | None = None,
    domain: ResearchDomain | str | None = None,
    api_key: str | None = None,
) -> ChatAgent:
    """Create a judge agent that evaluates evidence quality.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.
        api_key: Optional BYOK key (auto-detects provider from prefix).

    Returns:
        ChatAgent configured for evidence assessment
    """
    client = chat_client or get_chat_client(api_key=api_key)

    return ChatAgent(
        name="JudgeAgent",
        description="Evaluates evidence quality and determines if sufficient for synthesis",
        instructions=get_judge_prompt(domain),
        chat_client=client,
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )


def create_hypothesis_agent(
    chat_client: BaseChatClient | None = None,
    domain: ResearchDomain | str | None = None,
    api_key: str | None = None,
) -> ChatAgent:
    """Create a hypothesis generation agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.
        api_key: Optional BYOK key (auto-detects provider from prefix).

    Returns:
        ChatAgent configured for hypothesis generation
    """
    client = chat_client or get_chat_client(api_key=api_key)
    config = get_domain_config(domain)

    return ChatAgent(
        name="HypothesisAgent",
        description=config.hypothesis_agent_description,
        instructions=get_hypothesis_prompt(domain),
        chat_client=client,
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )


def create_report_agent(
    chat_client: BaseChatClient | None = None,
    domain: ResearchDomain | str | None = None,
    api_key: str | None = None,
) -> ChatAgent:
    """Create a report synthesis agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.
        api_key: Optional BYOK key (auto-detects provider from prefix).

    Returns:
        ChatAgent configured for report generation
    """
    client = chat_client or get_chat_client(api_key=api_key)

    return ChatAgent(
        name="ReportAgent",
        description="Synthesizes research findings into structured reports",
        instructions=get_report_prompt(domain),
        chat_client=client,
        tools=[get_bibliography],
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )
