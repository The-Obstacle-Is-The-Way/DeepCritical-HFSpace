"""Prompts for Report Agent."""

from typing import TYPE_CHECKING, Any

from src.config.domain import ResearchDomain, get_domain_config
from src.utils.text_utils import select_diverse_evidence, truncate_at_sentence

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol
    from src.utils.models import Evidence, HypothesisAssessment


def get_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for the report agent."""
    config = get_domain_config(domain)

    return f"""You are a scientific writer specializing in {config.name}.
Your role is to synthesize evidence into clear recommendations for interventions
with proper safety considerations.

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

Be comprehensive but concise. Cite evidence for all claims."""


# Keep SYSTEM_PROMPT for backwards compatibility
SYSTEM_PROMPT = get_system_prompt()


async def format_report_prompt(
    query: str,
    evidence: list["Evidence"],
    hypotheses: list["HypothesisAssessment"] | list[Any],
    assessment: Any,
    metadata: dict[str, Any],
    embeddings: "EmbeddingServiceProtocol | None" = None,
) -> str:
    """Format prompt for report generation.

    Args:
        query: Research query
        evidence: Collected evidence
        hypotheses: Generated hypotheses
        assessment: Judge assessment details
        metadata: Search metadata
        embeddings: Optional embedding service for diverse selection
    """
    # Select diverse evidence (max 15 for report)
    selected = await select_diverse_evidence(evidence, n=15, query=query, embeddings=embeddings)

    evidence_text = "\n".join(
        [
            f"- **{e.citation.title}** ({e.citation.source}): "
            f"{truncate_at_sentence(e.content, 400)}"
            for e in selected
        ]
    )

    # Format hypotheses if available
    hypotheses_text = "No specific hypotheses generated."
    if hypotheses:
        # Handle both Pydantic models and dicts/objects
        h_list = []
        for h in hypotheses:
            if hasattr(h, "hypotheses"):
                for item in h.hypotheses:
                    h_list.append(f"- {item.drug} -> {item.target} -> {item.effect}")
            elif isinstance(h, dict):
                h_list.append(str(h))
            else:
                h_list.append(str(h))
        if h_list:
            hypotheses_text = "\n".join(h_list)

    return f"""Generate a comprehensive research report for: "{query}""

## Context
- **Sources Searched**: {", ".join(metadata.get("sources", []))}
- **Iterations**: {metadata.get("iterations", 0)}

## Evidence ({len(selected)} key papers)
{evidence_text}

## Generated Hypotheses
{hypotheses_text}

## Task
Synthesize this information into a structured report following the Executive Summary format.
Focus on clinical applicability and safety.
Use specific citations from the evidence list."""
