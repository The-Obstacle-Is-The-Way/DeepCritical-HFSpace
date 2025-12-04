"""Prompts for Hypothesis Agent."""

from typing import TYPE_CHECKING

from src.config.domain import ResearchDomain, get_domain_config
from src.utils.text_utils import select_diverse_evidence, truncate_at_sentence

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol
    from src.utils.models import Evidence


def get_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for the hypothesis agent."""
    config = get_domain_config(domain)

    return f"""You are a biomedical research scientist specializing in {config.name}.
Your role is to generate evidence-based hypotheses for interventions,
identifying mechanisms of action and potential therapeutic applications.

Based on evidence:

1. Identify the key molecular targets involved
2. Map the biological pathways affected
3. Generate testable hypotheses in this format:

   DRUG -> TARGET -> PATHWAY -> THERAPEUTIC EFFECT

   Example:
   Testosterone -> Androgen receptor -> Dopamine modulation -> Enhanced libido

4. Explain the rationale for each hypothesis
5. Suggest what additional evidence would support or refute it

Focus on mechanistic plausibility and existing evidence."""


# Keep SYSTEM_PROMPT for backwards compatibility (used by PydanticAI agents)
SYSTEM_PROMPT = get_system_prompt()


async def format_hypothesis_prompt(
    query: str,
    evidence: list["Evidence"],
    embeddings: "EmbeddingServiceProtocol | None" = None,
) -> str:
    """Format prompt for hypothesis generation.

    Uses smart evidence selection instead of arbitrary truncation.

    Args:
        query: The research query
        evidence: All collected evidence
        embeddings: Optional EmbeddingService for diverse selection
    """
    # Select diverse, relevant evidence (not arbitrary first 10)
    # We use n=10 as a reasonable context window limit
    selected = await select_diverse_evidence(evidence, n=10, query=query, embeddings=embeddings)

    # Format with sentence-aware truncation
    evidence_text = "\n".join(
        [
            f"- **{e.citation.title}** ({e.citation.source}): "
            f"{truncate_at_sentence(e.content, 300)}"
            for e in selected
        ]
    )

    return f"""Based on the following evidence about "{query}", generate mechanistic hypotheses.

## Evidence ({len(selected)} papers selected for diversity)
{evidence_text}

## Task
1. Identify potential drug targets mentioned in the evidence
2. Propose mechanism hypotheses (Drug -> Target -> Pathway -> Effect)
3. Rate confidence based on evidence strength
4. Suggest searches to test each hypothesis

Generate 2-4 hypotheses, prioritized by confidence."""
