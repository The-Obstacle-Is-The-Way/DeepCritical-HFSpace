"""Prompts for Report Agent."""

from typing import TYPE_CHECKING, Any

from src.config.domain import ResearchDomain, get_domain_config
from src.utils.text_utils import select_diverse_evidence, truncate_at_sentence

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol
    from src.utils.models import Evidence, MechanismHypothesis


def get_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for the report agent."""
    config = get_domain_config(domain)
    return f"""{config.report_system_prompt}

Your role is to synthesize evidence and hypotheses into a clear, structured report.

A good report:
1. Has a clear EXECUTIVE SUMMARY (one paragraph, key takeaways)
2. States the RESEARCH QUESTION clearly
3. Describes METHODOLOGY (what was searched, how)
4. Evaluates HYPOTHESES with evidence counts
5. Separates MECHANISTIC and CLINICAL findings
6. Lists specific DRUG CANDIDATES
7. Acknowledges LIMITATIONS honestly
8. Provides a balanced CONCLUSION
9. Includes properly formatted REFERENCES

Write in scientific but accessible language. Be specific about evidence strength.

─────────────────────────────────────────────────────────────────────────────
🚨 CRITICAL: REQUIRED JSON STRUCTURE 🚨
─────────────────────────────────────────────────────────────────────────────

The `hypotheses_tested` field MUST be a LIST of objects, each with these fields:
- "hypothesis": the hypothesis text
- "supported": count of supporting evidence (integer)
- "contradicted": count of contradicting evidence (integer)

Example:
  hypotheses_tested: [
    {{"hypothesis": "Metformin -> AMPK -> reduced inflammation",
      "supported": 3, "contradicted": 1}},
    {{"hypothesis": "Aspirin inhibits COX-2 pathway",
      "supported": 5, "contradicted": 0}}
  ]

The `references` field MUST be a LIST of objects, each with these fields:
- "title": paper title (string)
- "authors": author names (string)
- "source": "pubmed" or "web" (string)
- "url": the EXACT URL from evidence (string)

Example:
  references: [
    {{"title": "Metformin and Cancer", "authors": "Smith et al.", "source": "pubmed", "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"}}
  ]

─────────────────────────────────────────────────────────────────────────────
🚨 CRITICAL CITATION REQUIREMENTS 🚨
─────────────────────────────────────────────────────────────────────────────

You MUST follow these rules for the References section:

1. You may ONLY cite papers that appear in the Evidence section above
2. Every reference URL must EXACTLY match a provided evidence URL
3. Do NOT invent, fabricate, or hallucinate any references
4. Do NOT modify paper titles, authors, dates, or URLs
5. If unsure about a citation, OMIT it rather than guess
6. Copy URLs exactly as provided - do not create similar-looking URLs

VIOLATION OF THESE RULES PRODUCES DANGEROUS MISINFORMATION.
─────────────────────────────────────────────────────────────────────────────"""


# Keep SYSTEM_PROMPT for backwards compatibility
SYSTEM_PROMPT = get_system_prompt()


async def format_report_prompt(
    query: str,
    evidence: list["Evidence"],
    hypotheses: list["MechanismHypothesis"],
    assessment: dict[str, Any],
    metadata: dict[str, Any],
    embeddings: "EmbeddingServiceProtocol | None" = None,
) -> str:
    """Format prompt for report generation.

    Includes full evidence details for accurate citation.
    """
    # Select diverse evidence (not arbitrary truncation)
    selected = await select_diverse_evidence(evidence, n=20, query=query, embeddings=embeddings)

    # Include FULL citation details for each evidence item
    # This helps the LLM create accurate references
    evidence_lines = []
    for e in selected:
        authors = ", ".join(e.citation.authors or ["Unknown"])
        evidence_lines.append(
            f"- **Title**: {e.citation.title}\n"
            f"  **URL**: {e.citation.url}\n"
            f"  **Authors**: {authors}\n"
            f"  **Date**: {e.citation.date or 'n.d.'}\n"
            f"  **Source**: {e.citation.source}\n"
            f"  **Content**: {truncate_at_sentence(e.content, 200)}\n"
        )
    evidence_summary = "\n".join(evidence_lines)

    if hypotheses:
        hypotheses_lines = []
        for h in hypotheses:
            hypotheses_lines.append(
                f"- {h.drug} -> {h.target} -> {h.pathway} -> {h.effect} "
                f"(Confidence: {h.confidence:.0%})"
            )
        hypotheses_summary = "\n".join(hypotheses_lines)
    else:
        hypotheses_summary = "No hypotheses generated yet."

    sources = ", ".join(metadata.get("sources", []))

    return f"""Generate a structured research report for the following query.

## Original Query
{query}

## Evidence Collected ({len(selected)} papers, selected for diversity)

{evidence_summary}

## Hypotheses Generated
{hypotheses_summary}

## Assessment Scores
- Mechanism Score: {assessment.get("mechanism_score", "N/A")}/10
- Clinical Evidence Score: {assessment.get("clinical_score", "N/A")}/10
- Overall Confidence: {assessment.get("confidence", 0):.0%}

## Metadata
- Sources Searched: {sources}
- Search Iterations: {metadata.get("iterations", 0)}

Generate a complete ResearchReport with all sections filled in.

REMINDER: Only cite papers from the Evidence section above. Copy URLs exactly."""
