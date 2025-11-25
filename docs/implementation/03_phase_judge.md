# Phase 3 Implementation Spec: Judge Vertical Slice

**Goal**: Implement the "Brain" of the agent — evaluating evidence quality.
**Philosophy**: "Structured Output or Bust."
**Estimated Effort**: 3-4 hours
**Prerequisite**: Phase 2 complete

---

## 1. The Slice Definition

This slice covers:
1. **Input**: Question + List of `Evidence`.
2. **Process**:
   - Construct prompt with evidence.
   - Call LLM (PydanticAI).
   - Parse into `JudgeAssessment`.
3. **Output**: `JudgeAssessment` object.

**Files**:
- `src/utils/models.py`: Add Judge models (DrugCandidate, JudgeAssessment)
- `src/prompts/judge.py`: Prompt templates
- `src/prompts/__init__.py`: Package init
- `src/agent_factory/judges.py`: Handler logic

---

## 2. Models (`src/utils/models.py`)

Add these to the existing models file (after SearchResult):

```python
# Add to src/utils/models.py (after SearchResult class)

class DrugCandidate(BaseModel):
    """A potential drug repurposing candidate identified from evidence."""

    drug_name: str = Field(description="Name of the drug")
    original_indication: str = Field(description="What the drug was originally approved for")
    proposed_indication: str = Field(description="The new condition it might treat")
    mechanism: str = Field(description="How it might work for the new indication")
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        description="Strength of evidence supporting this candidate"
    )


class JudgeAssessment(BaseModel):
    """The judge's assessment of evidence sufficiency."""

    sufficient: bool = Field(
        description="Whether we have enough evidence to synthesize a report"
    )
    recommendation: Literal["continue", "synthesize"] = Field(
        description="Whether to continue searching or synthesize a report"
    )
    reasoning: str = Field(
        description="Explanation of the assessment",
        min_length=10,
        max_length=1000
    )
    overall_quality_score: int = Field(
        ge=1, le=10,
        description="Overall quality of evidence (1-10)"
    )
    coverage_score: int = Field(
        ge=1, le=10,
        description="How well evidence covers the question (1-10)"
    )
    candidates: list[DrugCandidate] = Field(
        default_factory=list,
        description="Drug candidates identified from the evidence"
    )
    next_search_queries: list[str] = Field(
        default_factory=list,
        description="Suggested queries if more searching is needed"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Gaps in the current evidence"
    )
```

---

## 3. Prompts (`src/prompts/__init__.py`)

```python
"""Prompt templates package."""
from src.prompts.judge import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

__all__ = ["JUDGE_SYSTEM_PROMPT", "build_judge_user_prompt"]
```

---

## 4. Prompts (`src/prompts/judge.py`)

```python
"""Prompt templates for the Judge agent."""
from typing import List
from src.utils.models import Evidence


JUDGE_SYSTEM_PROMPT = """You are an expert biomedical research judge evaluating evidence for drug repurposing hypotheses.

Your role is to:
1. Assess the quality and relevance of retrieved evidence
2. Identify potential drug repurposing candidates
3. Determine if sufficient evidence exists to write a report
4. Suggest additional search queries if evidence is insufficient

Evaluation Criteria:
- **Quality**: Is the evidence from reputable sources (peer-reviewed journals, clinical trials)?
- **Relevance**: Does the evidence directly address the research question?
- **Recency**: Is the evidence recent (prefer last 5 years for clinical relevance)?
- **Diversity**: Do we have evidence from multiple independent sources?
- **Mechanism**: Is there a plausible biological mechanism?

Scoring Guidelines:
- Overall Quality (1-10): 1-3 = poor/unreliable, 4-6 = moderate, 7-10 = high quality
- Coverage (1-10): 1-3 = major gaps, 4-6 = partial coverage, 7-10 = comprehensive

Decision Rules:
- If quality >= 6 AND coverage >= 6 AND at least 1 drug candidate: recommend "synthesize"
- Otherwise: recommend "continue" and provide next_search_queries

Always identify drug candidates when evidence supports them, including:
- Drug name
- Original indication
- Proposed new indication
- Mechanism of action
- Evidence strength (weak/moderate/strong)

Be objective and scientific. Avoid speculation without evidence."""


def build_judge_user_prompt(question: str, evidence: List[Evidence]) -> str:
    """Build the user prompt for the judge.

    Args:
        question: The original research question.
        evidence: List of Evidence objects to evaluate.

    Returns:
        Formatted prompt string.
    """
    # Format evidence into readable blocks
    evidence_blocks = []
    for i, e in enumerate(evidence, 1):
        block = f"""
### Evidence {i}
**Source**: {e.citation.source.upper()}
**Title**: {e.citation.title}
**Date**: {e.citation.date}
**Authors**: {', '.join(e.citation.authors[:3]) or 'Unknown'}
**URL**: {e.citation.url}
**Relevance Score**: {e.relevance:.2f}

**Content**:
{e.content[:1500]}
"""
        evidence_blocks.append(block)

    evidence_text = "\n---\n".join(evidence_blocks) if evidence_blocks else "No evidence provided."

    return f"""## Research Question
{question}

## Retrieved Evidence ({len(evidence)} items)
{evidence_text}

## Your Task
Evaluate the evidence above and provide your assessment. Consider:
1. Is the evidence sufficient to answer the research question?
2. What drug repurposing candidates can be identified?
3. What gaps exist in the evidence?
4. Should we continue searching or synthesize a report?

Provide your assessment in the structured format."""


def build_synthesis_prompt(question: str, assessment: "JudgeAssessment", evidence: List[Evidence]) -> str:
    """Build the prompt for report synthesis.

    Args:
        question: The original research question.
        assessment: The judge's assessment.
        evidence: List of Evidence objects.

    Returns:
        Formatted prompt for synthesis.
    """
    candidates_text = ""
    if assessment.candidates:
        candidates_text = "\n## Identified Drug Candidates\n"
        for c in assessment.candidates:
            candidates_text += f"""
### {c.drug_name}
- **Original Use**: {c.original_indication}
- **Proposed Use**: {c.proposed_indication}
- **Mechanism**: {c.mechanism}
- **Evidence Strength**: {c.evidence_strength}
"""

    evidence_summary = "\n".join([
        f"- [{e.citation.source.upper()}] {e.citation.title} ({e.citation.date})"
        for e in evidence[:10]
    ])

    return f"""## Research Question
{question}

{candidates_text}

## Evidence Summary
{evidence_summary}

## Quality Assessment
- Overall Quality: {assessment.overall_quality_score}/10
- Coverage: {assessment.coverage_score}/10
- Reasoning: {assessment.reasoning}

## Your Task
Write a comprehensive research report summarizing the drug repurposing possibilities.
Include:
1. Executive Summary
2. Background on the condition
3. Drug candidates with evidence
4. Mechanisms of action
5. Current clinical trial status (if mentioned)
6. Recommendations for further research
7. References

Format as professional markdown suitable for researchers."""
```

---

## 5. Handler (`src/agent_factory/judges.py`)

```python
"""Judge handler - evaluates evidence quality using LLM."""
import structlog
from typing import List
from pydantic_ai import Agent
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.exceptions import JudgeError
from src.utils.models import JudgeAssessment, Evidence
from src.prompts.judge import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

logger = structlog.get_logger()


def _get_model_string() -> str:
    """Get the PydanticAI model string from settings.

    PydanticAI expects format like 'openai:gpt-4o-mini' or 'anthropic:claude-3-haiku-20240307'.
    """
    provider = settings.llm_provider
    model = settings.llm_model

    # If model already has provider prefix, return as-is
    if ":" in model:
        return model

    # Otherwise, prefix with provider
    return f"{provider}:{model}"


# Initialize the PydanticAI Agent for judging
# This uses structured output to guarantee JudgeAssessment schema
judge_agent = Agent(
    model=_get_model_string(),
    result_type=JudgeAssessment,
    system_prompt=JUDGE_SYSTEM_PROMPT,
)


class JudgeHandler:
    """Handles evidence assessment using LLM."""

    def __init__(self, agent: Agent | None = None):
        """Initialize the judge handler.

        Args:
            agent: Optional PydanticAI agent (for testing/mocking).
        """
        self.agent = agent or judge_agent

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def assess(self, question: str, evidence: List[Evidence]) -> JudgeAssessment:
        """Assess the quality and sufficiency of evidence.

        Args:
            question: The research question being investigated.
            evidence: List of Evidence objects to evaluate.

        Returns:
            JudgeAssessment with scores, candidates, and recommendation.

        Raises:
            JudgeError: If assessment fails after retries.
        """
        logger.info(
            "judge_assessment_starting",
            question=question[:100],
            evidence_count=len(evidence)
        )

        # Handle empty evidence case
        if not evidence:
            logger.warning("judge_no_evidence", question=question[:100])
            return JudgeAssessment(
                sufficient=False,
                recommendation="continue",
                reasoning="No evidence was provided to evaluate. Need to search for relevant research.",
                overall_quality_score=1,
                coverage_score=1,
                candidates=[],
                next_search_queries=[
                    f"{question} clinical trial",
                    f"{question} mechanism",
                    f"{question} drug repurposing",
                ],
                gaps=["No evidence collected yet"],
            )

        try:
            # Build the prompt
            prompt = build_judge_user_prompt(question, evidence)

            # Call the LLM with structured output
            result = await self.agent.run(prompt)

            logger.info(
                "judge_assessment_complete",
                sufficient=result.data.sufficient,
                recommendation=result.data.recommendation,
                quality_score=result.data.overall_quality_score,
                coverage_score=result.data.coverage_score,
                candidates_found=len(result.data.candidates),
            )

            return result.data

        except Exception as e:
            logger.error("judge_assessment_failed", error=str(e))
            raise JudgeError(f"Evidence assessment failed: {e}") from e

    async def should_continue(self, assessment: JudgeAssessment) -> bool:
        """Check if we should continue searching based on assessment.

        Args:
            assessment: The judge's assessment.

        Returns:
            True if we should search more, False if ready to synthesize.
        """
        return assessment.recommendation == "continue"
```

---

## 6. TDD Workflow

### Test File: `tests/unit/agent_factory/test_judges.py`

```python
"""Unit tests for JudgeHandler."""
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestJudgeHandler:
    """Tests for JudgeHandler."""

    @pytest.mark.asyncio
    async def test_assess_returns_assessment(self, mocker):
        """JudgeHandler.assess should return JudgeAssessment."""
        from src.agent_factory.judges import JudgeHandler
        from src.utils.models import JudgeAssessment, Evidence, Citation

        # Create mock assessment result
        mock_assessment = JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Good quality evidence from multiple sources.",
            overall_quality_score=8,
            coverage_score=7,
            candidates=[],
            next_search_queries=[],
            gaps=[],
        )

        # Mock PydanticAI agent result
        mock_result = MagicMock()
        mock_result.data = mock_assessment

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Create evidence
        evidence = [
            Evidence(
                content="Test evidence content about drug repurposing.",
                citation=Citation(
                    source="pubmed",
                    title="Test Article",
                    url="https://pubmed.ncbi.nlm.nih.gov/123/",
                    date="2024",
                    authors=["Smith J", "Jones K"],
                ),
                relevance=0.9,
            )
        ]

        handler = JudgeHandler(agent=mock_agent)
        result = await handler.assess("Can metformin treat Alzheimer's?", evidence)

        assert result.sufficient is True
        assert result.recommendation == "synthesize"
        assert result.overall_quality_score == 8
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_assess_handles_empty_evidence(self):
        """JudgeHandler should handle empty evidence gracefully."""
        from src.agent_factory.judges import JudgeHandler

        # Use real handler but don't call LLM
        handler = JudgeHandler()

        # Empty evidence should return default assessment
        result = await handler.assess("Test question?", [])

        assert result.sufficient is False
        assert result.recommendation == "continue"
        assert result.overall_quality_score == 1
        assert len(result.next_search_queries) > 0

    @pytest.mark.asyncio
    async def test_assess_with_drug_candidates(self, mocker):
        """JudgeHandler should identify drug candidates from evidence."""
        from src.agent_factory.judges import JudgeHandler
        from src.utils.models import JudgeAssessment, DrugCandidate, Evidence, Citation

        # Create assessment with candidates
        mock_assessment = JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Strong evidence for metformin.",
            overall_quality_score=8,
            coverage_score=8,
            candidates=[
                DrugCandidate(
                    drug_name="Metformin",
                    original_indication="Type 2 Diabetes",
                    proposed_indication="Alzheimer's Disease",
                    mechanism="Activates AMPK, reduces inflammation",
                    evidence_strength="moderate",
                )
            ],
            next_search_queries=[],
            gaps=[],
        )

        mock_result = MagicMock()
        mock_result.data = mock_assessment

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        evidence = [
            Evidence(
                content="Metformin shows neuroprotective properties...",
                citation=Citation(
                    source="pubmed",
                    title="Metformin and Alzheimer's",
                    url="https://pubmed.ncbi.nlm.nih.gov/456/",
                    date="2024",
                ),
            )
        ]

        handler = JudgeHandler(agent=mock_agent)
        result = await handler.assess("Can metformin treat Alzheimer's?", evidence)

        assert len(result.candidates) == 1
        assert result.candidates[0].drug_name == "Metformin"
        assert result.candidates[0].evidence_strength == "moderate"

    @pytest.mark.asyncio
    async def test_should_continue_returns_correct_value(self):
        """should_continue should return True for 'continue' recommendation."""
        from src.agent_factory.judges import JudgeHandler
        from src.utils.models import JudgeAssessment

        handler = JudgeHandler()

        # Test continue case
        continue_assessment = JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Need more evidence.",
            overall_quality_score=4,
            coverage_score=3,
        )
        assert await handler.should_continue(continue_assessment) is True

        # Test synthesize case
        synthesize_assessment = JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Sufficient evidence.",
            overall_quality_score=8,
            coverage_score=8,
        )
        assert await handler.should_continue(synthesize_assessment) is False

    @pytest.mark.asyncio
    async def test_assess_handles_llm_error(self, mocker):
        """JudgeHandler should raise JudgeError on LLM failure."""
        from src.agent_factory.judges import JudgeHandler
        from src.utils.models import Evidence, Citation
        from src.utils.exceptions import JudgeError

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))

        evidence = [
            Evidence(
                content="Test content",
                citation=Citation(
                    source="pubmed",
                    title="Test",
                    url="https://example.com",
                    date="2024",
                ),
            )
        ]

        handler = JudgeHandler(agent=mock_agent)

        with pytest.raises(JudgeError) as exc_info:
            await handler.assess("Test question?", evidence)

        assert "assessment failed" in str(exc_info.value).lower()


class TestPromptBuilding:
    """Tests for prompt building functions."""

    def test_build_judge_user_prompt_formats_evidence(self):
        """build_judge_user_prompt should format evidence correctly."""
        from src.prompts.judge import build_judge_user_prompt
        from src.utils.models import Evidence, Citation

        evidence = [
            Evidence(
                content="Metformin shows neuroprotective effects in animal models.",
                citation=Citation(
                    source="pubmed",
                    title="Metformin Neuroprotection Study",
                    url="https://pubmed.ncbi.nlm.nih.gov/123/",
                    date="2024-01-15",
                    authors=["Smith J", "Jones K", "Brown M"],
                ),
                relevance=0.85,
            )
        ]

        prompt = build_judge_user_prompt("Can metformin treat Alzheimer's?", evidence)

        # Check question is included
        assert "Can metformin treat Alzheimer's?" in prompt

        # Check evidence is formatted
        assert "PUBMED" in prompt
        assert "Metformin Neuroprotection Study" in prompt
        assert "2024-01-15" in prompt
        assert "Smith J" in prompt
        assert "0.85" in prompt  # Relevance score

    def test_build_judge_user_prompt_handles_empty_evidence(self):
        """build_judge_user_prompt should handle empty evidence."""
        from src.prompts.judge import build_judge_user_prompt

        prompt = build_judge_user_prompt("Test question?", [])

        assert "Test question?" in prompt
        assert "No evidence provided" in prompt
```

---

## 7. Implementation Checklist

- [ ] Add `DrugCandidate` and `JudgeAssessment` models to `src/utils/models.py`
- [ ] Create `src/prompts/__init__.py`
- [ ] Create `src/prompts/judge.py` (complete prompt templates)
- [ ] Implement `src/agent_factory/judges.py` (complete JudgeHandler class)
- [ ] Write tests in `tests/unit/agent_factory/test_judges.py`
- [ ] Run `uv run pytest tests/unit/agent_factory/ -v` — **ALL TESTS MUST PASS**
- [ ] Run `uv run ruff check src/agent_factory src/prompts` — **NO ERRORS**
- [ ] Run `uv run mypy src/agent_factory src/prompts` — **NO ERRORS**
- [ ] Commit: `git commit -m "feat: phase 3 judge slice complete"`

---

## 8. Definition of Done

Phase 3 is **COMPLETE** when:

1. ✅ All unit tests in `tests/unit/agent_factory/` pass
2. ✅ `JudgeHandler` returns valid `JudgeAssessment` objects
3. ✅ Structured output is enforced (no raw JSON strings leaked)
4. ✅ Retry/exception handling is covered by tests
5. ✅ Ruff and mypy pass with no errors
6. ✅ Manual REPL sanity check works (requires API key):

```python
import asyncio
from src.agent_factory.judges import JudgeHandler
from src.utils.models import Evidence, Citation

async def test():
    handler = JudgeHandler()
    evidence = [
        Evidence(
            content="Metformin shows neuroprotective properties in multiple studies. "
                    "AMPK activation reduces neuroinflammation and may slow cognitive decline.",
            citation=Citation(
                source="pubmed",
                title="Metformin and Cognitive Function: A Review",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                date="2024",
                authors=["Smith J", "Jones K"],
            ),
            relevance=0.9,
        )
    ]
    result = await handler.assess("Can metformin treat Alzheimer's?", evidence)
    print(f"Sufficient: {result.sufficient}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Quality: {result.overall_quality_score}/10")
    print(f"Coverage: {result.coverage_score}/10")
    print(f"Reasoning: {result.reasoning}")
    if result.candidates:
        print(f"Candidates: {[c.drug_name for c in result.candidates]}")

asyncio.run(test())
```

**Proceed to Phase 4 ONLY after all checkboxes are complete.**
