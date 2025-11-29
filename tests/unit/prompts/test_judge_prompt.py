from unittest.mock import patch

import pytest

from src.prompts.judge import format_user_prompt, select_evidence_for_judge
from src.utils.models import Citation, Evidence


def make_evidence(title: str, content: str = "content") -> Evidence:
    return Evidence(
        content=content,
        citation=Citation(title=title, url="http://test.com", date="2025", source="pubmed"),
    )


@pytest.mark.asyncio
async def test_evidence_selection_diverse():
    """Verify evidence selection includes early and recent items (fallback logic)."""
    # Create enough evidence to trigger selection
    evidence = [make_evidence(f"Paper {i}") for i in range(100)]

    # Mock select_diverse_evidence to raise ImportError to trigger fallback logic
    with patch("src.utils.text_utils.select_diverse_evidence", side_effect=ImportError):
        selected = await select_evidence_for_judge(evidence, "test query", max_items=30)

    assert len(selected) == 30

    # Should include some early evidence (lost-in-the-middle mitigation)
    titles = [e.citation.title for e in selected]

    # Check for start (Paper 0..9)
    has_early = any(f"Paper {i}" in title for title in titles for i in range(10))
    # Check for end (Paper 90..99)
    has_late = any(f"Paper {i}" in title for title in titles for i in range(90, 100))

    assert has_early, "Should include early evidence"
    assert has_late, "Should include recent evidence"


def test_prompt_includes_question_at_edges():
    """Verify lost-in-the-middle mitigation in prompt formatting."""
    evidence = [make_evidence("Test Paper")]
    question = "CRITICAL RESEARCH QUESTION"

    prompt = format_user_prompt(question, evidence, iteration=5, max_iterations=10)

    # Question should appear at START and END of prompt
    lines = prompt.split("\n")

    # Check start (first few lines)
    start_content = "\n".join(lines[:10])
    assert question in start_content

    # Check end (last few lines)
    end_content = "\n".join(lines[-10:])
    assert question in end_content
    assert "REMINDER: Original Question" in end_content
