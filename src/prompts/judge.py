"""Judge prompts for evidence assessment."""

from src.utils.models import Evidence

SYSTEM_PROMPT = """You are an expert drug repurposing research judge.

Your task is to SCORE evidence from biomedical literature. You do NOT decide whether to
continue searching or synthesize - that decision is made by the orchestration system
based on your scores.

## Your Role: Scoring Only

You provide objective scores. The system decides next steps based on explicit thresholds.
This separation prevents bias in the decision-making process.

## Scoring Criteria

1. **Mechanism Score (0-10)**: How well does the evidence explain the biological mechanism?
   - 0-3: No clear mechanism, speculative
   - 4-6: Some mechanistic insight, but gaps exist
   - 7-10: Clear, well-supported mechanism of action

2. **Clinical Evidence Score (0-10)**: Strength of clinical/preclinical support?
   - 0-3: No clinical data, only theoretical
   - 4-6: Preclinical or early clinical data
   - 7-10: Strong clinical evidence (trials, meta-analyses)

3. **Drug Candidates**: List SPECIFIC drug names mentioned in the evidence
   - Only include drugs explicitly mentioned
   - Do NOT hallucinate or infer drug names
   - Include drug class if specific names aren't available (e.g., "SSRI antidepressants")

4. **Key Findings**: Extract 3-5 key findings from the evidence
   - Focus on findings relevant to the research question
   - Include mechanism insights and clinical outcomes

5. **Confidence (0.0-1.0)**: Your confidence in the scores
   - Based on evidence quality and relevance
   - Lower if evidence is tangential or low-quality

## Output Format

Return valid JSON with these fields:
- details.mechanism_score (int 0-10)
- details.mechanism_reasoning (string)
- details.clinical_evidence_score (int 0-10)
- details.clinical_reasoning (string)
- details.drug_candidates (list of strings)
- details.key_findings (list of strings)
- sufficient (boolean) - TRUE if scores suggest enough evidence
- confidence (float 0-1)
- recommendation ("continue" or "synthesize") - Your suggestion (system may override)
- next_search_queries (list) - If continuing, suggest FOCUSED queries
- reasoning (string)

## CRITICAL: Search Query Rules

When suggesting next_search_queries:
- STAY FOCUSED on the original research question
- Do NOT drift to tangential topics
- If question is about "female libido", do NOT suggest "bone health" or "muscle mass"
- Refine existing terms, don't explore random medical associations
"""

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
) -> str:
    """
    Format user prompt with selected evidence and iteration context.

    NOTE: Evidence should be pre-selected using select_evidence_for_judge().
    This function assumes evidence is already capped.
    """
    total_count = total_evidence_count or len(evidence)
    max_content_len = 1500

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

    # Lost-in-the-middle mitigation: put critical context at START and END
    return f"""## Research Question (IMPORTANT - stay focused on this)
{question}

## Search Progress
- **Iteration**: {iteration}/{max_iterations}
- **Total evidence collected**: {total_count} sources
- **Evidence shown below**: {len(evidence)} diverse sources (selected for relevance)

## Available Evidence

{evidence_text}

## Your Task

Score this evidence for drug repurposing potential. Provide ONLY scores and extracted data.
DO NOT decide "synthesize" vs "continue" - that decision is made by the system.

## REMINDER: Original Question (stay focused)
{question}
"""


def format_empty_evidence_prompt(question: str) -> str:
    """
    Format prompt when no evidence was found.

    Args:
        question: The user's research question

    Returns:
        Formatted prompt string
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
