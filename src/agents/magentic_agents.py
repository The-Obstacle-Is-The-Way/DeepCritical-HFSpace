"""Magentic-compatible agents using ChatAgent pattern."""

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from src.agents.tools import (
    get_bibliography,
    search_clinical_trials,
    search_preprints,
    search_pubmed,
)
from src.config.domain import ResearchDomain, get_domain_config
from src.utils.config import settings


def create_search_agent(
    chat_client: OpenAIChatClient | None = None,
    domain: ResearchDomain | str | None = None,
) -> ChatAgent:
    """Create a search agent with internal LLM and search tools.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.

    Returns:
        ChatAgent configured for biomedical search
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,  # Use configured model
        api_key=settings.openai_api_key,
    )
    config = get_domain_config(domain)

    return ChatAgent(
        name="SearchAgent",
        description=config.search_agent_description,
        instructions=f"""You are a biomedical search specialist. When asked to find evidence:

1. Analyze the request to determine what to search for
2. Extract key search terms (drug names, disease names, mechanisms)
3. Use the appropriate search tools:
   - search_pubmed for peer-reviewed papers
   - search_clinical_trials for clinical studies
   - search_preprints for cutting-edge findings
4. Summarize what you found and highlight key evidence

Be thorough - search multiple databases when appropriate.
Focus on finding: mechanisms of action, clinical evidence, and specific findings
related to {config.name}.""",
        chat_client=client,
        tools=[search_pubmed, search_clinical_trials, search_preprints],
        temperature=1.0,  # Explicitly set for reasoning model compatibility (o1/o3)
    )


def create_judge_agent(
    chat_client: OpenAIChatClient | None = None,
    domain: ResearchDomain | str | None = None,
) -> ChatAgent:
    """Create a judge agent that evaluates evidence quality.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.

    Returns:
        ChatAgent configured for evidence assessment
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    config = get_domain_config(domain)

    return ChatAgent(
        name="JudgeAgent",
        description="Evaluates evidence quality and determines if sufficient for synthesis",
        instructions=f"""{config.judge_system_prompt}

When asked to evaluate:

1. Review all evidence presented in the conversation
2. Score on two dimensions (0-10 each):
   - Mechanism Score: How well is the biological mechanism explained?
   - Clinical Score: How strong is the clinical/preclinical evidence?
3. Determine if evidence is SUFFICIENT for a final report:
   - Sufficient: Clear mechanism + supporting clinical data
   - Insufficient: Gaps in mechanism OR weak clinical evidence
4. If insufficient, suggest specific search queries to fill gaps

Be rigorous but fair. Look for:
- Molecular targets and pathways
- Animal model studies
- Human clinical trials
- Safety data
- Drug-drug interactions""",
        chat_client=client,
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )


def create_hypothesis_agent(
    chat_client: OpenAIChatClient | None = None,
    domain: ResearchDomain | str | None = None,
) -> ChatAgent:
    """Create a hypothesis generation agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.

    Returns:
        ChatAgent configured for hypothesis generation
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    config = get_domain_config(domain)

    return ChatAgent(
        name="HypothesisAgent",
        description=config.hypothesis_agent_description,
        instructions=f"""{config.hypothesis_system_prompt}

Based on evidence:

1. Identify the key molecular targets involved
2. Map the biological pathways affected
3. Generate testable hypotheses in this format:

   DRUG -> TARGET -> PATHWAY -> THERAPEUTIC EFFECT

   Example:
   Metformin -> AMPK activation -> mTOR inhibition -> Reduced tau phosphorylation

4. Explain the rationale for each hypothesis
5. Suggest what additional evidence would support or refute it

Focus on mechanistic plausibility and existing evidence.""",
        chat_client=client,
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )


def create_report_agent(
    chat_client: OpenAIChatClient | None = None,
    domain: ResearchDomain | str | None = None,
) -> ChatAgent:
    """Create a report synthesis agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.
        domain: Research domain for customization.

    Returns:
        ChatAgent configured for report generation
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    config = get_domain_config(domain)

    return ChatAgent(
        name="ReportAgent",
        description="Synthesizes research findings into structured reports",
        instructions=f"""{config.report_system_prompt}

When asked to synthesize:

Generate a structured report with these sections:

## Executive Summary
Brief overview of findings and recommendation

## Methodology
Databases searched, queries used, evidence reviewed

## Key Findings
### Mechanism of Action
- Molecular targets
- Biological pathways
- Proposed mechanism

### Clinical Evidence
- Preclinical studies
- Clinical trials
- Safety profile

## Candidates
List specific candidates with potential

## Limitations
Gaps in evidence, conflicting data, caveats

## Conclusion
Final recommendation with confidence level

## References
Use the 'get_bibliography' tool to fetch the complete list of citations.
Format them as a numbered list.

Be comprehensive but concise. Cite evidence for all claims.""",
        chat_client=client,
        tools=[get_bibliography],
        temperature=1.0,  # Explicitly set for reasoning model compatibility
    )
