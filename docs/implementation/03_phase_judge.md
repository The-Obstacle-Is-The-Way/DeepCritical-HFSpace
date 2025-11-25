# Phase 3 Implementation Spec: Judge Vertical Slice

**Goal**: Implement the "Brain" of the agent — evaluating evidence quality and deciding next steps.
**Philosophy**: "Structured Output or Bust."
**Estimated Effort**: 3-4 hours
**Prerequisite**: Phase 2 complete (Search slice working)

---

## 1. The Slice Definition

This slice covers:
1. **Input**: A user question + a list of `Evidence` (from Phase 2).
2. **Process**:
   - Construct a prompt with the evidence.
   - Call LLM via **PydanticAI** (enforces structured output).
   - Parse response into typed assessment.
3. **Output**: A `JudgeAssessment` object with decision + next queries.

**Directory**: `src/features/judge/`

---

## 2. Why PydanticAI for the Judge?

We use **PydanticAI** because:
- ✅ **Structured Output**: Forces LLM to return valid JSON matching our Pydantic model
- ✅ **Retry Logic**: Built-in retry with exponential backoff
- ✅ **Multi-Provider**: Works with OpenAI, Anthropic, Gemini
- ✅ **Type Safety**: Full typing support

```python
# PydanticAI forces the LLM to return EXACTLY this structure
class JudgeAssessment(BaseModel):
    sufficient: bool
    recommendation: Literal["continue", "synthesize"]
    next_search_queries: list[str]
```

---

## 3. Models (`src/features/judge/models.py`)

```python
"""Data models for the Judge feature."""
from pydantic import BaseModel, Field
from typing import Literal


class EvidenceQuality(BaseModel):
    """Quality assessment of a single piece of evidence."""

    relevance_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How relevant is this evidence to the query (0-10)"
    )
    credibility_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How credible is the source (0-10)"
    )
    key_finding: str = Field(
        ...,
        max_length=200,
        description="One-sentence summary of the key finding"
    )


class DrugCandidate(BaseModel):
    """A potential drug repurposing candidate identified in the evidence."""

    drug_name: str = Field(..., description="Name of the drug")
    original_indication: str = Field(..., description="What the drug was originally approved for")
    proposed_indication: str = Field(..., description="The new proposed use")
    mechanism: str = Field(..., description="Proposed mechanism of action")
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        ...,
        description="Strength of supporting evidence"
    )


class JudgeAssessment(BaseModel):
    """The judge's assessment of the collected evidence."""

    # Core Decision
    sufficient: bool = Field(
        ...,
        description="Is there enough evidence to write a report?"
    )
    recommendation: Literal["continue", "synthesize"] = Field(
        ...,
        description="Should we search more or synthesize a report?"
    )

    # Reasoning
    reasoning: str = Field(
        ...,
        max_length=500,
        description="Explanation of the assessment"
    )

    # Scores
    overall_quality_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Overall quality of evidence (0-10)"
    )
    coverage_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How well does evidence cover the query (0-10)"
    )

    # Extracted Information
    candidates: list[DrugCandidate] = Field(
        default_factory=list,
        description="Drug candidates identified in the evidence"
    )

    # Next Steps (only if recommendation == "continue")
    next_search_queries: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Suggested follow-up queries if more evidence needed"
    )

    # Gaps Identified
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps identified in current evidence"
    )
```

---

## 4. Prompts (`src/features/judge/prompts.py`)

Prompts are **code**. They are versioned, tested, and parameterized.

```python
"""Prompt templates for the Judge feature."""
from typing import List
from src.features.search.models import Evidence


# System prompt - defines the judge's role and constraints
JUDGE_SYSTEM_PROMPT = """You are a biomedical research quality assessor specializing in drug repurposing.

Your job is to evaluate evidence retrieved from PubMed and web searches, and decide if:
1. There is SUFFICIENT evidence to write a research report
2. More searching is needed to fill gaps

## Evaluation Criteria

### For "sufficient" = True (ready to synthesize):
- At least 3 relevant pieces of evidence
- At least one peer-reviewed source (PubMed)
- Clear mechanism of action identified
- Drug candidates with at least "moderate" evidence strength

### For "sufficient" = False (continue searching):
- Fewer than 3 relevant pieces
- No clear drug candidates identified
- Major gaps in mechanism understanding
- All evidence is low quality

## Output Requirements
- Be STRICT. Only mark sufficient=True if evidence is genuinely adequate
- Always provide reasoning for your decision
- If continuing, suggest SPECIFIC, ACTIONABLE search queries
- Identify concrete gaps, not vague statements

## Important
- You are assessing DRUG REPURPOSING potential
- Focus on: mechanism of action, existing clinical data, safety profile
- Ignore marketing content or non-scientific sources"""


def format_evidence_for_prompt(evidence_list: List[Evidence]) -> str:
    """Format evidence list into a string for the prompt."""
    if not evidence_list:
        return "NO EVIDENCE COLLECTED YET"

    formatted = []
    for i, ev in enumerate(evidence_list, 1):
        formatted.append(f"""
--- Evidence #{i} ---
Source: {ev.citation.source.upper()}
Title: {ev.citation.title}
Date: {ev.citation.date}
URL: {ev.citation.url}

Content:
{ev.content[:1500]}
---""")

    return "\n".join(formatted)


def build_judge_user_prompt(question: str, evidence: List[Evidence]) -> str:
    """Build the user prompt for the judge."""
    evidence_text = format_evidence_for_prompt(evidence)

    return f"""## Research Question
{question}

## Collected Evidence ({len(evidence)} pieces)
{evidence_text}

## Your Task
Assess the evidence above and provide your structured assessment.
If evidence is insufficient, suggest 2-3 specific follow-up search queries."""


# For testing: a simplified prompt that's easier to mock
JUDGE_TEST_PROMPT = "Assess the following evidence and return a JudgeAssessment."
```

---

## 5. Handler (`src/features/judge/handlers.py`)

The handler uses **PydanticAI** for structured LLM output.

```python
"""Judge handler - evaluates evidence quality using LLM."""
from typing import List
import structlog
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.shared.config import settings
from src.shared.exceptions import JudgeError
from src.features.search.models import Evidence
from .models import JudgeAssessment
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

logger = structlog.get_logger()


def get_llm_model():
    """Get the configured LLM model for PydanticAI."""
    if settings.llm_provider == "openai":
        return OpenAIModel(
            settings.llm_model,
            api_key=settings.get_api_key(),
        )
    elif settings.llm_provider == "anthropic":
        return AnthropicModel(
            settings.llm_model,
            api_key=settings.get_api_key(),
        )
    else:
        raise JudgeError(f"Unknown LLM provider: {settings.llm_provider}")


# Create the PydanticAI agent with structured output
judge_agent = Agent(
    model=get_llm_model(),
    result_type=JudgeAssessment,  # Forces structured output!
    system_prompt=JUDGE_SYSTEM_PROMPT,
)


class JudgeHandler:
    """Handles evidence assessment using LLM."""

    def __init__(self, agent: Agent | None = None):
        """
        Initialize the judge handler.

        Args:
            agent: Optional PydanticAI agent (for testing injection)
        """
        self.agent = agent or judge_agent
        self._call_count = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def assess(
        self,
        question: str,
        evidence: List[Evidence],
    ) -> JudgeAssessment:
        """
        Assess the quality and sufficiency of evidence.

        Args:
            question: The original research question
            evidence: List of Evidence objects to assess

        Returns:
            JudgeAssessment with decision and recommendations

        Raises:
            JudgeError: If assessment fails after retries
        """
        logger.info(
            "Starting evidence assessment",
            question=question[:100],
            evidence_count=len(evidence),
        )

        self._call_count += 1

        # Build the prompt
        user_prompt = build_judge_user_prompt(question, evidence)

        try:
            # Run the agent - PydanticAI handles structured output
            result = await self.agent.run(user_prompt)

            # result.data is already a JudgeAssessment (typed!)
            assessment = result.data

            logger.info(
                "Assessment complete",
                sufficient=assessment.sufficient,
                recommendation=assessment.recommendation,
                quality_score=assessment.overall_quality_score,
                candidates_found=len(assessment.candidates),
            )

            return assessment

        except Exception as e:
            logger.error("Judge assessment failed", error=str(e))
            raise JudgeError(f"Failed to assess evidence: {e}") from e

    @property
    def call_count(self) -> int:
        """Number of LLM calls made (for budget tracking)."""
        return self._call_count


# Alternative: Direct OpenAI client (if PydanticAI doesn't work)
class FallbackJudgeHandler:
    """Fallback handler using direct OpenAI client with JSON mode."""

    def __init__(self):
        import openai
        self.client = openai.AsyncOpenAI(api_key=settings.get_api_key())

    async def assess(
        self,
        question: str,
        evidence: List[Evidence],
    ) -> JudgeAssessment:
        """Assess using direct OpenAI API with JSON mode."""
        from .prompts import build_judge_user_prompt

        user_prompt = build_judge_user_prompt(question, evidence)

        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent assessments
        )

        # Parse the JSON response
        import json
        content = response.choices[0].message.content
        data = json.loads(content)

        return JudgeAssessment.model_validate(data)
```

---

## 6. TDD Workflow

### Test File: `tests/unit/features/judge/test_handler.py`

```python
"""Unit tests for the Judge handler."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestJudgeModels:
    """Tests for Judge data models."""

    def test_judge_assessment_valid(self):
        """JudgeAssessment should accept valid data."""
        from src.features.judge.models import JudgeAssessment

        assessment = JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Strong evidence from multiple PubMed sources.",
            overall_quality_score=8,
            coverage_score=7,
            candidates=[],
            next_search_queries=[],
            gaps=[],
        )

        assert assessment.sufficient is True
        assert assessment.recommendation == "synthesize"

    def test_judge_assessment_score_bounds(self):
        """JudgeAssessment should reject invalid scores."""
        from src.features.judge.models import JudgeAssessment
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgeAssessment(
                sufficient=True,
                recommendation="synthesize",
                reasoning="Test",
                overall_quality_score=15,  # Invalid: > 10
                coverage_score=5,
            )

    def test_drug_candidate_model(self):
        """DrugCandidate should validate properly."""
        from src.features.judge.models import DrugCandidate

        candidate = DrugCandidate(
            drug_name="Metformin",
            original_indication="Type 2 Diabetes",
            proposed_indication="Alzheimer's Disease",
            mechanism="Reduces neuroinflammation via AMPK activation",
            evidence_strength="moderate",
        )

        assert candidate.drug_name == "Metformin"
        assert candidate.evidence_strength == "moderate"


class TestJudgePrompts:
    """Tests for prompt formatting."""

    def test_format_evidence_empty(self):
        """format_evidence_for_prompt should handle empty list."""
        from src.features.judge.prompts import format_evidence_for_prompt

        result = format_evidence_for_prompt([])
        assert "NO EVIDENCE" in result

    def test_format_evidence_with_items(self):
        """format_evidence_for_prompt should format evidence correctly."""
        from src.features.judge.prompts import format_evidence_for_prompt
        from src.features.search.models import Evidence, Citation

        evidence = [
            Evidence(
                content="Test content about metformin",
                citation=Citation(
                    source="pubmed",
                    title="Test Article",
                    url="https://pubmed.ncbi.nlm.nih.gov/123/",
                    date="2024-01-15",
                ),
            )
        ]

        result = format_evidence_for_prompt(evidence)

        assert "Evidence #1" in result
        assert "PUBMED" in result
        assert "Test Article" in result
        assert "metformin" in result

    def test_build_judge_user_prompt(self):
        """build_judge_user_prompt should include question and evidence."""
        from src.features.judge.prompts import build_judge_user_prompt
        from src.features.search.models import Evidence, Citation

        evidence = [
            Evidence(
                content="Sample content",
                citation=Citation(
                    source="pubmed",
                    title="Sample",
                    url="https://example.com",
                    date="2024",
                ),
            )
        ]

        result = build_judge_user_prompt(
            "What drugs could treat Alzheimer's?",
            evidence,
        )

        assert "Alzheimer" in result
        assert "1 pieces" in result


class TestJudgeHandler:
    """Tests for JudgeHandler."""

    @pytest.mark.asyncio
    async def test_assess_returns_assessment(self, mocker):
        """JudgeHandler.assess should return JudgeAssessment."""
        from src.features.judge.handlers import JudgeHandler
        from src.features.judge.models import JudgeAssessment
        from src.features.search.models import Evidence, Citation

        # Create a mock agent
        mock_result = MagicMock()
        mock_result.data = JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Good evidence",
            overall_quality_score=8,
            coverage_score=7,
        )

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Create handler with mock agent
        handler = JudgeHandler(agent=mock_agent)

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

        # Act
        result = await handler.assess("Test question", evidence)

        # Assert
        assert isinstance(result, JudgeAssessment)
        assert result.sufficient is True
        assert result.recommendation == "synthesize"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_assess_increments_call_count(self, mocker):
        """JudgeHandler should track LLM call count."""
        from src.features.judge.handlers import JudgeHandler
        from src.features.judge.models import JudgeAssessment

        mock_result = MagicMock()
        mock_result.data = JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Need more evidence",
            overall_quality_score=4,
            coverage_score=3,
            next_search_queries=["metformin mechanism"],
        )

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        handler = JudgeHandler(agent=mock_agent)

        assert handler.call_count == 0

        await handler.assess("Q1", [])
        assert handler.call_count == 1

        await handler.assess("Q2", [])
        assert handler.call_count == 2

    @pytest.mark.asyncio
    async def test_assess_raises_judge_error_on_failure(self, mocker):
        """JudgeHandler should raise JudgeError on failure."""
        from src.features.judge.handlers import JudgeHandler
        from src.shared.exceptions import JudgeError

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))

        handler = JudgeHandler(agent=mock_agent)

        with pytest.raises(JudgeError, match="Failed to assess"):
            await handler.assess("Test", [])

    @pytest.mark.asyncio
    async def test_assess_continues_when_insufficient(self, mocker):
        """JudgeHandler should return next_search_queries when insufficient."""
        from src.features.judge.handlers import JudgeHandler
        from src.features.judge.models import JudgeAssessment

        mock_result = MagicMock()
        mock_result.data = JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Not enough peer-reviewed sources",
            overall_quality_score=3,
            coverage_score=2,
            next_search_queries=[
                "metformin alzheimer clinical trial",
                "AMPK neuroprotection mechanism",
            ],
            gaps=["No clinical trial data", "Mechanism unclear"],
        )

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        handler = JudgeHandler(agent=mock_agent)
        result = await handler.assess("Test", [])

        assert result.sufficient is False
        assert result.recommendation == "continue"
        assert len(result.next_search_queries) == 2
        assert len(result.gaps) == 2
```

---

## 7. Integration Test (Optional, Real LLM)

```python
# tests/integration/test_judge_live.py
"""Integration tests that hit real LLM APIs (run manually)."""
import pytest
import os


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_judge_live_assessment():
    """Test real LLM assessment (requires API key)."""
    from src.features.judge.handlers import JudgeHandler
    from src.features.search.models import Evidence, Citation

    handler = JudgeHandler()

    evidence = [
        Evidence(
            content="""Metformin, a first-line antidiabetic drug, has shown
            neuroprotective properties in preclinical studies. The drug activates
            AMPK, which may reduce neuroinflammation and improve mitochondrial
            function in neurons.""",
            citation=Citation(
                source="pubmed",
                title="Metformin and Neuroprotection: A Review",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2024-01-15",
            ),
        ),
        Evidence(
            content="""A retrospective cohort study found that diabetic patients
            taking metformin had a 30% lower risk of developing dementia compared
            to those on other antidiabetic medications.""",
            citation=Citation(
                source="pubmed",
                title="Metformin Use and Dementia Risk",
                url="https://pubmed.ncbi.nlm.nih.gov/67890/",
                date="2023-11-20",
            ),
        ),
    ]

    result = await handler.assess(
        "What is the potential of metformin for treating Alzheimer's disease?",
        evidence,
    )

    # Basic sanity checks
    assert result.sufficient in [True, False]
    assert result.recommendation in ["continue", "synthesize"]
    assert 0 <= result.overall_quality_score <= 10
    assert len(result.reasoning) > 0


# Run with: uv run pytest tests/integration -m integration
```

---

## 8. Module Exports (`src/features/judge/__init__.py`)

```python
"""Judge feature - evidence quality assessment."""
from .models import JudgeAssessment, DrugCandidate, EvidenceQuality
from .handlers import JudgeHandler
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

__all__ = [
    "JudgeAssessment",
    "DrugCandidate",
    "EvidenceQuality",
    "JudgeHandler",
    "JUDGE_SYSTEM_PROMPT",
    "build_judge_user_prompt",
]
```

---

## 9. Implementation Checklist

- [ ] Create `src/features/judge/models.py` with all Pydantic models
- [ ] Create `src/features/judge/prompts.py` with prompt templates
- [ ] Create `src/features/judge/handlers.py` with `JudgeHandler`
- [ ] Create `src/features/judge/__init__.py` with exports
- [ ] Write tests in `tests/unit/features/judge/test_handler.py`
- [ ] Run `uv run pytest tests/unit/features/judge/ -v` — **ALL TESTS MUST PASS**
- [ ] (Optional) Run integration test with real API key
- [ ] Commit: `git commit -m "feat: phase 3 judge slice complete"`

---

## 10. Definition of Done

Phase 3 is **COMPLETE** when:

1. ✅ All unit tests pass
2. ✅ `JudgeHandler` returns valid `JudgeAssessment` objects
3. ✅ Structured output is enforced (no raw JSON strings)
4. ✅ Retry logic works (test by mocking transient failures)
5. ✅ Can run this in Python REPL (with API key):

```python
import asyncio
from src.features.judge.handlers import JudgeHandler
from src.features.search.models import Evidence, Citation

async def test():
    handler = JudgeHandler()
    evidence = [
        Evidence(
            content="Metformin shows neuroprotective properties...",
            citation=Citation(
                source="pubmed",
                title="Metformin Review",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                date="2024",
            ),
        )
    ]
    result = await handler.assess("Can metformin treat Alzheimer's?", evidence)
    print(f"Sufficient: {result.sufficient}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Reasoning: {result.reasoning}")

asyncio.run(test())
```

**Proceed to Phase 4 ONLY after all checkboxes are complete.**
