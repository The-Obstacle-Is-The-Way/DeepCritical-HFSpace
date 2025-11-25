"""Judge prompts for evidence assessment."""

from src.utils.models import Evidence

SYSTEM_PROMPT = """You are an expert drug repurposing research judge.

Your task is to evaluate evidence from biomedical literature and determine if it's sufficient to
recommend drug candidates for a given condition.

## Evaluation Criteria

1. **Mechanism Score (0-10)**: How well does the evidence explain the biological mechanism?
   - 0-3: No clear mechanism, speculative
   - 4-6: Some mechanistic insight, but gaps exist
   - 7-10: Clear, well-supported mechanism of action

2. **Clinical Evidence Score (0-10)**: Strength of clinical/preclinical support?
   - 0-3: No clinical data, only theoretical
   - 4-6: Preclinical or early clinical data
   - 7-10: Strong clinical evidence (trials, meta-analyses)

3. **Sufficiency**: Evidence is sufficient when:
   - Combined scores >= 12 AND
   - At least one specific drug candidate identified AND
   - Clear mechanistic rationale exists

## Output Rules

- Always output valid JSON matching the schema
- Be conservative: only recommend "synthesize" when truly confident
- If continuing, suggest specific, actionable search queries
- Never hallucinate drug names or findings not in the evidence
"""


def format_user_prompt(question: str, evidence: list[Evidence]) -> str:
    """
    Format the user prompt with question and evidence.

    Args:
        question: The user's research question
        evidence: List of Evidence objects from search

    Returns:
        Formatted prompt string
    """
    max_content_len = 1500

    def format_single_evidence(i: int, e: Evidence) -> str:
        content = e.content
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        return (
            f"### Evidence {i + 1}\n"
            f"**Source**: {e.citation.source.upper()} - {e.citation.title}\n"
            f"**URL**: {e.citation.url}\n"
            f"**Date**: {e.citation.date}\n"
            f"**Content**:\n{content}"
        )

    evidence_text = "\n\n".join([format_single_evidence(i, e) for i, e in enumerate(evidence)])

    return f"""## Research Question
{question}

## Available Evidence ({len(evidence)} sources)

{evidence_text}

## Your Task

Evaluate this evidence and determine if it's sufficient to recommend drug repurposing candidates.
Respond with a JSON object matching the JudgeAssessment schema.
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
