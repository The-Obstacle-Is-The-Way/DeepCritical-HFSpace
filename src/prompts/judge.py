"""Judge prompts for evidence assessment."""

from src.config.domain import ResearchDomain, get_domain_config
from src.utils.models import Evidence


def get_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for the judge agent (Magentic/Advanced Mode)."""
    config = get_domain_config(domain)

    return f"""You are an expert research judge specializing in {config.name}.
Your role is to evaluate evidence for interventions, assess efficacy and safety data,
and determine clinical applicability.

When asked to evaluate:

1. Review all evidence presented in the conversation
2. Score on two dimensions (0-10 each):
   - Mechanism Score: How well is the biological mechanism explained?
   - Clinical Score: How strong is the clinical/preclinical evidence?
3. Determine if evidence is SUFFICIENT for a final report:
   - Sufficient: Clear mechanism + supporting clinical data
   - Insufficient: Gaps in mechanism OR weak clinical evidence
4. If insufficient, suggest specific search queries to fill gaps

## CRITICAL OUTPUT FORMAT
To ensure the workflow terminates when appropriate, you MUST follow these rules:

IF evidence is SUFFICIENT (confidence >= 70%):
   Start your response with a line like:
   "✅ SUFFICIENT EVIDENCE (confidence: 72%). STOP SEARCHING. Delegate to ReportAgent NOW."
   Use your actual numeric confidence instead of 72.
   Then explain why.

IF evidence is INSUFFICIENT:
   Start with "❌ INSUFFICIENT: <Reason>."
   Then provide scores and next queries.

Be rigorous but fair. Look for:
- Molecular targets and pathways
- Animal model studies
- Human clinical trials
- Safety data
- Drug-drug interactions"""


def get_scoring_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the scoring instructions for the judge."""
    return """Score this evidence for relevance.
Provide ONLY scores and extracted data."""


# Keep SYSTEM_PROMPT for backwards compatibility
SYSTEM_PROMPT = get_system_prompt()

MAX_EVIDENCE_FOR_JUDGE = 30  # Keep under token limits


async def select_evidence_for_judge(
    evidence: list[Evidence],
    query: str,
    max_items: int = MAX_EVIDENCE_FOR_JUDGE,
) -> list[Evidence]:
    """
    Select diverse, relevant evidence for judge evaluation.

    Implements RAG best practices:
    - Diversity selection over recency-only
    - Lost-in-the-middle mitigation
    - Relevance re-ranking
    """
    if len(evidence) <= max_items:
        return evidence

    try:
        from src.utils.text_utils import select_diverse_evidence

        # Use embedding-based diversity selection
        return await select_diverse_evidence(evidence, n=max_items, query=query)
    except ImportError:
        # Fallback: mix of recent + early (lost-in-the-middle mitigation)
        early = evidence[: max_items // 3]  # First third
        recent = evidence[-(max_items * 2 // 3) :]  # Last two-thirds
        return early + recent


def format_user_prompt(
    question: str,
    evidence: list[Evidence],
    iteration: int = 0,
    max_iterations: int = 10,
    total_evidence_count: int | None = None,
    domain: ResearchDomain | str | None = None,
) -> str:
    """
    Format user prompt with selected evidence and iteration context.
    """
    # Use explicit None check - 0 is a valid count (empty evidence)
    total_count = total_evidence_count if total_evidence_count is not None else len(evidence)
    max_content_len = 1500
    scoring_prompt = get_scoring_prompt(domain)

    def format_single_evidence(i: int, e: Evidence) -> str:
        content = e.content
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        return (
            f"### Evidence {i + 1}\n"
            f"**Source**: {e.citation.source.upper()} - {e.citation.title}\n"
            f"**URL**: {e.citation.url}\n"
            f"**Content**:\n{content}"
        )

    evidence_text = "\n\n".join([format_single_evidence(i, e) for i, e in enumerate(evidence)])

    return f"""## Research Question (IMPORTANT - stay focused on this)
{question}

## Search Progress
- **Iteration**: {iteration}/{max_iterations}
- **Total evidence collected**: {total_count} sources
- **Evidence shown below**: {len(evidence)} diverse sources (selected for relevance)

## Available Evidence

{evidence_text}

## Your Task

{scoring_prompt}
"""


def format_empty_evidence_prompt(question: str) -> str:
    """
    Format prompt when no evidence was found.
    """
    return f"""## Research Question
{question}

## Available Evidence

No evidence was found from the search.

## Your Task

Since no evidence was found, recommend search queries that might yield better results.
Set sufficient=False and recommendation=\"continue\".
Suggest 3-5 specific search queries.
"""
