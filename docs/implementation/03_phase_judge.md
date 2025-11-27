# Phase 3 Implementation Spec: Judge Vertical Slice

**Goal**: Implement the "Brain" of the agent — evaluating evidence quality.
**Philosophy**: "Structured Output or Bust."
**Prerequisite**: Phase 2 complete (all search tests passing)

---

## 1. The Slice Definition

This slice covers:
1. **Input**: A user question + a list of `Evidence` (from Phase 2).
2. **Process**:
   - Construct a prompt with the evidence.
   - Call LLM (PydanticAI / OpenAI / Anthropic).
   - Force JSON structured output.
3. **Output**: A `JudgeAssessment` object.

**Files to Create**:
- `src/utils/models.py` - Add JudgeAssessment models (extend from Phase 2)
- `src/prompts/judge.py` - Judge prompt templates
- `src/agent_factory/judges.py` - JudgeHandler with PydanticAI
- `tests/unit/agent_factory/test_judges.py` - Unit tests

---

## 2. Models (Add to `src/utils/models.py`)

The output schema must be strict for reliable structured output.

```python
"""Add these models to src/utils/models.py (after Evidence models from Phase 2)."""
from pydantic import BaseModel, Field
from typing import List, Literal


class AssessmentDetails(BaseModel):
    """Detailed assessment of evidence quality."""

    mechanism_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How well does the evidence explain the mechanism? 0-10"
    )
    mechanism_reasoning: str = Field(
        ...,
        min_length=10,
        description="Explanation of mechanism score"
    )
    clinical_evidence_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Strength of clinical/preclinical evidence. 0-10"
    )
    clinical_reasoning: str = Field(
        ...,
        min_length=10,
        description="Explanation of clinical evidence score"
    )
    drug_candidates: List[str] = Field(
        default_factory=list,
        description="List of specific drug candidates mentioned"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from the evidence"
    )


class JudgeAssessment(BaseModel):
    """Complete assessment from the Judge."""

    details: AssessmentDetails
    sufficient: bool = Field(
        ...,
        description="Is evidence sufficient to provide a recommendation?"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment (0-1)"
    )
    recommendation: Literal["continue", "synthesize"] = Field(
        ...,
        description="continue = need more evidence, synthesize = ready to answer"
    )
    next_search_queries: List[str] = Field(
        default_factory=list,
        description="If continue, what queries to search next"
    )
    reasoning: str = Field(
        ...,
        min_length=20,
        description="Overall reasoning for the recommendation"
    )
```

---

## 3. Prompt Engineering (`src/prompts/judge.py`)

We treat prompts as code. They should be versioned and clean.

```python
"""Judge prompts for evidence assessment."""
from typing import List
from src.utils.models import Evidence


SYSTEM_PROMPT = """You are an expert drug repurposing research judge.

Your task is to evaluate evidence from biomedical literature and determine if it's sufficient to recommend drug candidates for a given condition.

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


def format_user_prompt(question: str, evidence: List[Evidence]) -> str:
    """
    Format the user prompt with question and evidence.

    Args:
        question: The user's research question
        evidence: List of Evidence objects from search

    Returns:
        Formatted prompt string
    """
    evidence_text = "\n\n".join([
        f"### Evidence {i+1}\n"
        f"**Source**: {e.citation.source.upper()} - {e.citation.title}\n"
        f"**URL**: {e.citation.url}\n"
        f"**Date**: {e.citation.date}\n"
        f"**Content**:\n{e.content[:1500]}..."
        if len(e.content) > 1500 else
        f"### Evidence {i+1}\n"
        f"**Source**: {e.citation.source.upper()} - {e.citation.title}\n"
        f"**URL**: {e.citation.url}\n"
        f"**Date**: {e.citation.date}\n"
        f"**Content**:\n{e.content}"
        for i, e in enumerate(evidence)
    ])

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
Set sufficient=False and recommendation="continue".
Suggest 3-5 specific search queries.
"""
```

---

## 4. JudgeHandler Implementation (`src/agent_factory/judges.py`)

Using PydanticAI for structured output with retry logic.

```python
"""Judge handler for evidence assessment using PydanticAI."""
import os
from typing import List
import structlog
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

from src.utils.models import Evidence, JudgeAssessment, AssessmentDetails
from src.utils.config import settings
from src.prompts.judge import SYSTEM_PROMPT, format_user_prompt, format_empty_evidence_prompt

logger = structlog.get_logger()


def get_model():
    """Get the LLM model based on configuration."""
    provider = getattr(settings, "llm_provider", "openai")

    if provider == "anthropic":
        return AnthropicModel(
            model_name=getattr(settings, "anthropic_model", "claude-3-5-sonnet-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        return OpenAIModel(
            model_name=getattr(settings, "openai_model", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )


class JudgeHandler:
    """
    Handles evidence assessment using an LLM with structured output.

    Uses PydanticAI to ensure responses match the JudgeAssessment schema.
    """

    def __init__(self, model=None):
        """
        Initialize the JudgeHandler.

        Args:
            model: Optional PydanticAI model. If None, uses config default.
        """
        self.model = model or get_model()
        self.agent = Agent(
            model=self.model,
            result_type=JudgeAssessment,
            system_prompt=SYSTEM_PROMPT,
            retries=3,
        )

    async def assess(
        self,
        question: str,
        evidence: List[Evidence],
    ) -> JudgeAssessment:
        """
        Assess evidence and determine if it's sufficient.

        Args:
            question: The user's research question
            evidence: List of Evidence objects from search

        Returns:
            JudgeAssessment with evaluation results

        Raises:
            JudgeError: If assessment fails after retries
        """
        logger.info(
            "Starting evidence assessment",
            question=question[:100],
            evidence_count=len(evidence),
        )

        # Format the prompt based on whether we have evidence
        if evidence:
            user_prompt = format_user_prompt(question, evidence)
        else:
            user_prompt = format_empty_evidence_prompt(question)

        try:
            # Run the agent with structured output
            result = await self.agent.run(user_prompt)
            assessment = result.data

            logger.info(
                "Assessment complete",
                sufficient=assessment.sufficient,
                recommendation=assessment.recommendation,
                confidence=assessment.confidence,
            )

            return assessment

        except Exception as e:
            logger.error("Assessment failed", error=str(e))
            # Return a safe default assessment on failure
            return self._create_fallback_assessment(question, str(e))

    def _create_fallback_assessment(
        self,
        question: str,
        error: str,
    ) -> JudgeAssessment:
        """
        Create a fallback assessment when LLM fails.

        Args:
            question: The original question
            error: The error message

        Returns:
            Safe fallback JudgeAssessment
        """
        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="Assessment failed due to LLM error",
                clinical_evidence_score=0,
                clinical_reasoning="Assessment failed due to LLM error",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=[
                f"{question} mechanism",
                f"{question} clinical trials",
                f"{question} drug candidates",
            ],
            reasoning=f"Assessment failed: {error}. Recommend retrying with refined queries.",
        )


class HFInferenceJudgeHandler:
    """
    JudgeHandler using HuggingFace Inference API for FREE LLM calls.

    This is the DEFAULT for demo mode - provides real AI analysis without
    requiring users to have OpenAI/Anthropic API keys.

    Model Fallback Chain (handles gated models and rate limits):
        1. meta-llama/Llama-3.1-8B-Instruct (best quality, requires HF_TOKEN)
        2. mistralai/Mistral-7B-Instruct-v0.3 (good quality, may require token)
        3. HuggingFaceH4/zephyr-7b-beta (ungated, always works)

    Rate Limit Handling:
        - Exponential backoff with 3 retries
        - Falls back to next model on persistent 429/503 errors
    """

    # Model fallback chain: gated (best) → ungated (fallback)
    FALLBACK_MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",      # Best quality (gated)
        "mistralai/Mistral-7B-Instruct-v0.3",    # Good quality
        "HuggingFaceH4/zephyr-7b-beta",          # Ungated fallback
    ]

    def __init__(self, model_id: str | None = None) -> None:
        """
        Initialize with HF Inference client.

        Args:
            model_id: Optional specific model ID. If None, uses FALLBACK_MODELS chain.
        """
        self.model_id = model_id
        # Will automatically use HF_TOKEN from env if available
        self.client = InferenceClient()
        self.call_count = 0
        self.last_question: str | None = None
        self.last_evidence: list[Evidence] | None = None

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """
        Robust JSON extraction that handles markdown blocks and nested braces.
        """
        text = text.strip()

        # Remove markdown code blocks if present (with bounds checking)
        if "```json" in text:
            parts = text.split("```json", 1)
            if len(parts) > 1:
                inner_parts = parts[1].split("```", 1)
                text = inner_parts[0]
        elif "```" in text:
            parts = text.split("```", 1)
            if len(parts) > 1:
                inner_parts = parts[1].split("```", 1)
                text = inner_parts[0]

        text = text.strip()

        # Find first '{'
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        # Stack-based parsing ignoring chars in strings
        count = 0
        in_string = False
        escape = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                count += 1
            elif char == "}":
                count -= 1
                if count == 0:
                    try:
                        result = json.loads(text[start_idx : i + 1])
                        if isinstance(result, dict):
                            return result
                        return None
                    except json.JSONDecodeError:
                        return None

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_with_retry(self, model: str, prompt: str, question: str) -> JudgeAssessment:
        """Make API call with retry logic using chat_completion."""
        loop = asyncio.get_running_loop()

        # Build messages for chat_completion (model-agnostic)
        messages = [
            {
                "role": "system",
                "content": f"""{SYSTEM_PROMPT}

IMPORTANT: Respond with ONLY valid JSON matching this schema:
{{
    "details": {{
        "mechanism_score": <int 0-10>,
        "mechanism_reasoning": "<string>",
        "clinical_evidence_score": <int 0-10>,
        "clinical_reasoning": "<string>",
        "drug_candidates": ["<string>", ...],
        "key_findings": ["<string>", ...]
    }},
    "sufficient": <bool>,
    "confidence": <float 0-1>,
    "recommendation": "continue" | "synthesize",
    "next_search_queries": ["<string>", ...],
    "reasoning": "<string>"
}}""",
            },
            {"role": "user", "content": prompt},
        ]

        # Use chat_completion (conversational task - supported by all models)
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=1024,
                temperature=0.1,
            ),
        )

        # Extract content from response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        # Extract and parse JSON
        json_data = self._extract_json(content)
        if not json_data:
            raise ValueError("No valid JSON found in response")

        return JudgeAssessment(**json_data)

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> JudgeAssessment:
        """
        Assess evidence using HuggingFace Inference API.
        Attempts models in order until one succeeds.
        """
        self.call_count += 1
        self.last_question = question
        self.last_evidence = evidence

        # Format the user prompt
        if evidence:
            user_prompt = format_user_prompt(question, evidence)
        else:
            user_prompt = format_empty_evidence_prompt(question)

        models_to_try: list[str] = [self.model_id] if self.model_id else self.FALLBACK_MODELS
        last_error: Exception | None = None

        for model in models_to_try:
            try:
                return await self._call_with_retry(model, user_prompt, question)
            except Exception as e:
                logger.warning("Model failed", model=model, error=str(e))
                last_error = e
                continue

        # All models failed
        logger.error("All HF models failed", error=str(last_error))
        return self._create_fallback_assessment(question, str(last_error))

    def _create_fallback_assessment(
        self,
        question: str,
        error: str,
    ) -> JudgeAssessment:
        """Create a fallback assessment when inference fails."""
        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning=f"Assessment failed: {error}",
                clinical_evidence_score=0,
                clinical_reasoning=f"Assessment failed: {error}",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=[
                f"{question} mechanism",
                f"{question} clinical trials",
                f"{question} drug candidates",
            ],
            reasoning=f"HF Inference failed: {error}. Recommend retrying.",
        )


class MockJudgeHandler:
    """
    Mock JudgeHandler for UNIT TESTING ONLY.

    NOT for production use. Use HFInferenceJudgeHandler for demo mode.
    """

    def __init__(self, mock_response: JudgeAssessment | None = None):
        """Initialize with optional mock response for testing."""
        self.mock_response = mock_response
        self.call_count = 0
        self.last_question = None
        self.last_evidence = None

    async def assess(
        self,
        question: str,
        evidence: List[Evidence],
    ) -> JudgeAssessment:
        """Return the mock response (for testing only)."""
        self.call_count += 1
        self.last_question = question
        self.last_evidence = evidence

        if self.mock_response:
            return self.mock_response

        # Default mock response for tests
        return JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=7,
                mechanism_reasoning="Mock assessment for testing",
                clinical_evidence_score=6,
                clinical_reasoning="Mock assessment for testing",
                drug_candidates=["TestDrug"],
                key_findings=["Test finding"],
            ),
            sufficient=len(evidence) >= 3,
            confidence=0.75,
            recommendation="synthesize" if len(evidence) >= 3 else "continue",
            next_search_queries=["query 1", "query 2"] if len(evidence) < 3 else [],
            reasoning="Mock assessment for unit testing only",
        )
```

---

## 5. TDD Workflow

### Test File: `tests/unit/agent_factory/test_judges.py`

```python
"""Unit tests for JudgeHandler."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.models import (
    Evidence,
    Citation,
    JudgeAssessment,
    AssessmentDetails,
)


class TestJudgeHandler:
    """Tests for JudgeHandler."""

    @pytest.mark.asyncio
    async def test_assess_returns_assessment(self):
        """JudgeHandler should return JudgeAssessment from LLM."""
        from src.agent_factory.judges import JudgeHandler

        # Create mock assessment
        mock_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=8,
                mechanism_reasoning="Strong mechanistic evidence",
                clinical_evidence_score=7,
                clinical_reasoning="Good clinical support",
                drug_candidates=["Metformin"],
                key_findings=["Neuroprotective effects"],
            ),
            sufficient=True,
            confidence=0.85,
            recommendation="synthesize",
            next_search_queries=[],
            reasoning="Evidence is sufficient for synthesis",
        )

        # Mock the PydanticAI agent
        mock_result = MagicMock()
        mock_result.data = mock_assessment

        with patch("src.agent_factory.judges.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            # Replace the agent with our mock
            handler.agent = mock_agent

            evidence = [
                Evidence(
                    content="Metformin shows neuroprotective properties...",
                    citation=Citation(
                        source="pubmed",
                        title="Metformin in AD",
                        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                        date="2024-01-01",
                    ),
                )
            ]

            result = await handler.assess("metformin alzheimer", evidence)

            assert result.sufficient is True
            assert result.recommendation == "synthesize"
            assert result.confidence == 0.85
            assert "Metformin" in result.details.drug_candidates

    @pytest.mark.asyncio
    async def test_assess_empty_evidence(self):
        """JudgeHandler should handle empty evidence gracefully."""
        from src.agent_factory.judges import JudgeHandler

        mock_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="No evidence to assess",
                clinical_evidence_score=0,
                clinical_reasoning="No evidence to assess",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=["metformin alzheimer mechanism"],
            reasoning="No evidence found, need to search more",
        )

        mock_result = MagicMock()
        mock_result.data = mock_assessment

        with patch("src.agent_factory.judges.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            handler.agent = mock_agent

            result = await handler.assess("metformin alzheimer", [])

            assert result.sufficient is False
            assert result.recommendation == "continue"
            assert len(result.next_search_queries) > 0

    @pytest.mark.asyncio
    async def test_assess_handles_llm_failure(self):
        """JudgeHandler should return fallback on LLM failure."""
        from src.agent_factory.judges import JudgeHandler

        with patch("src.agent_factory.judges.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(side_effect=Exception("API Error"))
            mock_agent_class.return_value = mock_agent

            handler = JudgeHandler()
            handler.agent = mock_agent

            evidence = [
                Evidence(
                    content="Some content",
                    citation=Citation(
                        source="pubmed",
                        title="Title",
                        url="url",
                        date="2024",
                    ),
                )
            ]

            result = await handler.assess("test question", evidence)

            # Should return fallback, not raise
            assert result.sufficient is False
            assert result.recommendation == "continue"
            assert "failed" in result.reasoning.lower()


class TestHFInferenceJudgeHandler:
    """Tests for HFInferenceJudgeHandler."""

    @pytest.mark.asyncio
    async def test_extract_json_raw(self):
        """Should extract raw JSON."""
        from src.agent_factory.judges import HFInferenceJudgeHandler

        handler = HFInferenceJudgeHandler.__new__(HFInferenceJudgeHandler)
        # Bypass __init__ for unit testing extraction

        result = handler._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_extract_json_markdown_block(self):
        """Should extract JSON from markdown code block."""
        from src.agent_factory.judges import HFInferenceJudgeHandler

        handler = HFInferenceJudgeHandler.__new__(HFInferenceJudgeHandler)

        response = '''Here is the assessment:
```json
{"key": "value", "nested": {"inner": 1}}
```
'''
        result = handler._extract_json(response)
        assert result == {"key": "value", "nested": {"inner": 1}}

    @pytest.mark.asyncio
    async def test_extract_json_with_preamble(self):
        """Should extract JSON with preamble text."""
        from src.agent_factory.judges import HFInferenceJudgeHandler

        handler = HFInferenceJudgeHandler.__new__(HFInferenceJudgeHandler)

        response = 'Here is your JSON response:\n{"sufficient": true, "confidence": 0.85}'
        result = handler._extract_json(response)
        assert result == {"sufficient": True, "confidence": 0.85}

    @pytest.mark.asyncio
    async def test_extract_json_nested_braces(self):
        """Should handle nested braces correctly."""
        from src.agent_factory.judges import HFInferenceJudgeHandler

        handler = HFInferenceJudgeHandler.__new__(HFInferenceJudgeHandler)

        response = '{"details": {"mechanism_score": 8}, "reasoning": "test"}'
        result = handler._extract_json(response)
        assert result["details"]["mechanism_score"] == 8

    @pytest.mark.asyncio
    async def test_hf_handler_uses_fallback_models(self):
        """HFInferenceJudgeHandler should have fallback model chain."""
        from src.agent_factory.judges import HFInferenceJudgeHandler

        # Check class has fallback models defined
        assert len(HFInferenceJudgeHandler.FALLBACK_MODELS) >= 3
        assert "zephyr-7b-beta" in HFInferenceJudgeHandler.FALLBACK_MODELS[-1]

    @pytest.mark.asyncio
    async def test_hf_handler_fallback_on_auth_error(self):
        """Should fall back to ungated model on auth error."""
        from src.agent_factory.judges import HFInferenceJudgeHandler
        from unittest.mock import MagicMock, patch

        with patch("src.agent_factory.judges.InferenceClient") as mock_client_class:
            # First call raises 403, second succeeds
            mock_client = MagicMock()
            mock_client.chat_completion.side_effect = [
                Exception("403 Forbidden: gated model"),
                MagicMock(choices=[MagicMock(message=MagicMock(content='{"sufficient": false}'))])
            ]
            mock_client_class.return_value = mock_client

            handler = HFInferenceJudgeHandler()
            # Manually trigger fallback test
            assert handler._try_fallback_model() is True
            assert handler.model_id != "meta-llama/Llama-3.1-8B-Instruct"


class TestMockJudgeHandler:
    """Tests for MockJudgeHandler (UNIT TESTING ONLY)."""

    @pytest.mark.asyncio
    async def test_mock_handler_returns_default(self):
        """MockJudgeHandler should return default assessment."""
        from src.agent_factory.judges import MockJudgeHandler

        handler = MockJudgeHandler()

        evidence = [
            Evidence(
                content="Content 1",
                citation=Citation(source="pubmed", title="T1", url="u1", date="2024"),
            ),
            Evidence(
                content="Content 2",
                citation=Citation(source="web", title="T2", url="u2", date="2024"),
            ),
        ]

        result = await handler.assess("test", evidence)

        assert handler.call_count == 1
        assert handler.last_question == "test"
        assert len(handler.last_evidence) == 2
        assert result.details.mechanism_score == 7

    @pytest.mark.asyncio
    async def test_mock_handler_custom_response(self):
        """MockJudgeHandler should return custom response when provided."""
        from src.agent_factory.judges import MockJudgeHandler

        custom_assessment = JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=10,
                mechanism_reasoning="Custom reasoning",
                clinical_evidence_score=10,
                clinical_reasoning="Custom clinical",
                drug_candidates=["CustomDrug"],
                key_findings=["Custom finding"],
            ),
            sufficient=True,
            confidence=1.0,
            recommendation="synthesize",
            next_search_queries=[],
            reasoning="Custom assessment",
        )

        handler = MockJudgeHandler(mock_response=custom_assessment)
        result = await handler.assess("test", [])

        assert result.details.mechanism_score == 10
        assert result.details.drug_candidates == ["CustomDrug"]

    @pytest.mark.asyncio
    async def test_mock_handler_insufficient_with_few_evidence(self):
        """MockJudgeHandler should recommend continue with < 3 evidence."""
        from src.agent_factory.judges import MockJudgeHandler

        handler = MockJudgeHandler()

        # Only 2 pieces of evidence
        evidence = [
            Evidence(
                content="Content",
                citation=Citation(source="pubmed", title="T", url="u", date="2024"),
            ),
            Evidence(
                content="Content 2",
                citation=Citation(source="web", title="T2", url="u2", date="2024"),
            ),
        ]

        result = await handler.assess("test", evidence)

        assert result.sufficient is False
        assert result.recommendation == "continue"
        assert len(result.next_search_queries) > 0
```

---

## 6. Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing deps ...
    "pydantic-ai>=0.0.16",
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "huggingface-hub>=0.20.0",  # For HFInferenceJudgeHandler (FREE LLM)
]
```

**Note**: `huggingface-hub` is required for the free tier to work. It:
- Provides `InferenceClient` for API calls
- Auto-reads `HF_TOKEN` from environment (optional, for gated models)
- Works without any token for ungated models like `zephyr-7b-beta`

---

## 7. Configuration (`src/utils/config.py`)

Add LLM configuration:

```python
"""Add to src/utils/config.py."""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings."""

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # API Keys (loaded from environment)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ncbi_api_key: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

---

## 8. Implementation Checklist

- [ ] Add `AssessmentDetails` and `JudgeAssessment` models to `src/utils/models.py`
- [ ] Create `src/prompts/__init__.py` (empty, for package)
- [ ] Create `src/prompts/judge.py` with prompt templates
- [ ] Create `src/agent_factory/__init__.py` with exports
- [ ] Implement `src/agent_factory/judges.py` with JudgeHandler
- [ ] Update `src/utils/config.py` with LLM settings
- [ ] Create `tests/unit/agent_factory/__init__.py`
- [ ] Write tests in `tests/unit/agent_factory/test_judges.py`
- [ ] Run `uv run pytest tests/unit/agent_factory/ -v` — **ALL TESTS MUST PASS**
- [ ] Commit: `git commit -m "feat: phase 3 judge slice complete"`

---

## 9. Definition of Done

Phase 3 is **COMPLETE** when:

1. All unit tests pass: `uv run pytest tests/unit/agent_factory/ -v`
2. `JudgeHandler` can assess evidence and return structured output
3. Graceful degradation: if LLM fails, returns safe fallback
4. MockJudgeHandler works for testing without API calls
5. Can run this in Python REPL:

```python
import asyncio
import os
from src.utils.models import Evidence, Citation
from src.agent_factory.judges import JudgeHandler, MockJudgeHandler

# Test with mock (no API key needed)
async def test_mock():
    handler = MockJudgeHandler()
    evidence = [
        Evidence(
            content="Metformin shows neuroprotective effects in AD models",
            citation=Citation(
                source="pubmed",
                title="Metformin and Alzheimer's",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2024-01-01",
            ),
        ),
    ]
    result = await handler.assess("metformin alzheimer", evidence)
    print(f"Sufficient: {result.sufficient}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Drug candidates: {result.details.drug_candidates}")

asyncio.run(test_mock())

# Test with real LLM (requires API key)
async def test_real():
    os.environ["OPENAI_API_KEY"] = "your-key-here"  # Or set in .env
    handler = JudgeHandler()
    evidence = [
        Evidence(
            content="Metformin shows neuroprotective effects in AD models...",
            citation=Citation(
                source="pubmed",
                title="Metformin and Alzheimer's",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2024-01-01",
            ),
        ),
    ]
    result = await handler.assess("metformin alzheimer", evidence)
    print(f"Sufficient: {result.sufficient}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

# asyncio.run(test_real())  # Uncomment with valid API key
```

**Proceed to Phase 4 ONLY after all checkboxes are complete.**
