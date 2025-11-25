# Phase 7 Implementation Spec: Hypothesis Agent

**Goal**: Add an agent that generates scientific hypotheses to guide targeted searches.
**Philosophy**: "Don't just find evidence—understand the mechanisms."
**Prerequisite**: Phase 6 complete (Embeddings working)

---

## 1. Why Hypothesis Agent?

Current limitation: **Search is reactive, not hypothesis-driven.**

Current flow:
1. User asks about "metformin alzheimer"
2. Search finds papers
3. Judge says "need more evidence"
4. Search again with slightly different keywords

With Hypothesis Agent:
1. User asks about "metformin alzheimer"
2. Search finds initial papers
3. **Hypothesis Agent analyzes**: "Evidence suggests metformin → AMPK activation → autophagy → amyloid clearance"
4. Search can now target: "metformin AMPK", "autophagy neurodegeneration", "amyloid clearance drugs"

**Key insight**: Scientific research is hypothesis-driven. The agent should think like a researcher.

---

## 2. Architecture

### Current (Phase 6)
```
User Query → Magentic Manager
                ├── SearchAgent → Evidence
                └── JudgeAgent → Sufficient? → Synthesize/Continue
```

### Phase 7
```
User Query → Magentic Manager
                ├── SearchAgent → Evidence
                ├── HypothesisAgent → Mechanistic Hypotheses  ← NEW
                └── JudgeAgent → Sufficient? → Synthesize/Continue
                       ↑
                  Uses hypotheses to guide next search
```

### Shared Context Enhancement
```python
evidence_store = {
    "current": [],
    "embeddings": {},
    "vector_index": None,
    "hypotheses": [],        # NEW: Generated hypotheses
    "tested_hypotheses": [], # NEW: Hypotheses with supporting/contradicting evidence
}
```

---

## 3. Hypothesis Model

### 3.1 Data Model (`src/utils/models.py`)

```python
class MechanismHypothesis(BaseModel):
    """A scientific hypothesis about drug mechanism."""

    drug: str = Field(description="The drug being studied")
    target: str = Field(description="Molecular target (e.g., AMPK, mTOR)")
    pathway: str = Field(description="Biological pathway affected")
    effect: str = Field(description="Downstream effect on disease")
    confidence: float = Field(ge=0, le=1, description="Confidence in hypothesis")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="PMIDs or URLs supporting this hypothesis"
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="PMIDs or URLs contradicting this hypothesis"
    )
    search_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested searches to test this hypothesis"
    )

    def to_search_queries(self) -> list[str]:
        """Generate search queries to test this hypothesis."""
        return [
            f"{self.drug} {self.target}",
            f"{self.target} {self.pathway}",
            f"{self.pathway} {self.effect}",
            *self.search_suggestions
        ]
```

### 3.2 Hypothesis Assessment

```python
class HypothesisAssessment(BaseModel):
    """Assessment of evidence against hypotheses."""

    hypotheses: list[MechanismHypothesis]
    primary_hypothesis: MechanismHypothesis | None = Field(
        description="Most promising hypothesis based on current evidence"
    )
    knowledge_gaps: list[str] = Field(
        description="What we don't know yet"
    )
    recommended_searches: list[str] = Field(
        description="Searches to fill knowledge gaps"
    )
```

---

## 4. Implementation

### 4.0 Text Utilities (`src/utils/text_utils.py`)

> **Why These Utilities?**
>
> The original spec used arbitrary truncation (`evidence[:10]` and `content[:300]`).
> This loses important information randomly. These utilities provide:
> 1. **Sentence-aware truncation** - cuts at sentence boundaries, not mid-word
> 2. **Diverse evidence selection** - uses embeddings to select varied evidence (MMR)

```python
"""Text processing utilities for evidence handling."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService
    from src.utils.models import Evidence


def truncate_at_sentence(text: str, max_chars: int = 300) -> str:
    """Truncate text at sentence boundary, preserving meaning.

    Args:
        text: The text to truncate
        max_chars: Maximum characters (default 300)

    Returns:
        Text truncated at last complete sentence within limit
    """
    if len(text) <= max_chars:
        return text

    # Find truncation point
    truncated = text[:max_chars]

    # Look for sentence endings: . ! ? followed by space or end
    for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars // 2:  # Don't truncate too aggressively
            return text[:last_sep + 1].strip()

    # Fallback: find last period
    last_period = truncated.rfind('.')
    if last_period > max_chars // 2:
        return text[:last_period + 1].strip()

    # Last resort: truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return text[:last_space].strip() + "..."

    return truncated + "..."


async def select_diverse_evidence(
    evidence: list["Evidence"],
    n: int,
    query: str,
    embeddings: "EmbeddingService | None" = None
) -> list["Evidence"]:
    """Select n most diverse and relevant evidence items.

    Uses Maximal Marginal Relevance (MMR) when embeddings available,
    falls back to relevance_score sorting otherwise.

    Args:
        evidence: All available evidence
        n: Number of items to select
        query: Original query for relevance scoring
        embeddings: Optional EmbeddingService for semantic diversity

    Returns:
        Selected evidence items, diverse and relevant
    """
    if not evidence:
        return []

    if n >= len(evidence):
        return evidence

    # Fallback: sort by relevance score if no embeddings
    if embeddings is None:
        return sorted(
            evidence,
            key=lambda e: e.relevance_score,
            reverse=True
        )[:n]

    # MMR: Maximal Marginal Relevance for diverse selection
    # Score = λ * relevance - (1-λ) * max_similarity_to_selected
    lambda_param = 0.7  # Balance relevance vs diversity

    # Get query embedding
    query_emb = await embeddings.embed(query)

    # Get all evidence embeddings
    evidence_embs = await embeddings.embed_batch([e.content for e in evidence])

    # Compute relevance scores (cosine similarity to query)
    from numpy import dot
    from numpy.linalg import norm
    cosine = lambda a, b: float(dot(a, b) / (norm(a) * norm(b)))

    relevance_scores = [cosine(query_emb, emb) for emb in evidence_embs]

    # Greedy MMR selection
    selected_indices: list[int] = []
    remaining = set(range(len(evidence)))

    for _ in range(n):
        best_score = float('-inf')
        best_idx = -1

        for idx in remaining:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component: max similarity to already selected
            if selected_indices:
                max_sim = max(
                    cosine(evidence_embs[idx], evidence_embs[sel])
                    for sel in selected_indices
                )
            else:
                max_sim = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [evidence[i] for i in selected_indices]
```

### 4.1 Hypothesis Prompts (`src/prompts/hypothesis.py`)

```python
"""Prompts for Hypothesis Agent."""
from src.utils.text_utils import truncate_at_sentence, select_diverse_evidence

SYSTEM_PROMPT = """You are a biomedical research scientist specializing in drug repurposing.

Your role is to generate mechanistic hypotheses based on evidence.

A good hypothesis:
1. Proposes a MECHANISM: Drug → Target → Pathway → Effect
2. Is TESTABLE: Can be supported or refuted by literature search
3. Is SPECIFIC: Names actual molecular targets and pathways
4. Generates SEARCH QUERIES: Helps find more evidence

Example hypothesis format:
- Drug: Metformin
- Target: AMPK (AMP-activated protein kinase)
- Pathway: mTOR inhibition → autophagy activation
- Effect: Enhanced clearance of amyloid-beta in Alzheimer's
- Confidence: 0.7
- Search suggestions: ["metformin AMPK brain", "autophagy amyloid clearance"]

Be specific. Use actual gene/protein names when possible."""


async def format_hypothesis_prompt(
    query: str,
    evidence: list,
    embeddings=None
) -> str:
    """Format prompt for hypothesis generation.

    Uses smart evidence selection instead of arbitrary truncation.

    Args:
        query: The research query
        evidence: All collected evidence
        embeddings: Optional EmbeddingService for diverse selection
    """
    # Select diverse, relevant evidence (not arbitrary first 10)
    selected = await select_diverse_evidence(
        evidence, n=10, query=query, embeddings=embeddings
    )

    # Format with sentence-aware truncation
    evidence_text = "\n".join([
        f"- **{e.citation.title}** ({e.citation.source}): {truncate_at_sentence(e.content, 300)}"
        for e in selected
    ])

    return f"""Based on the following evidence about "{query}", generate mechanistic hypotheses.

## Evidence ({len(selected)} papers selected for diversity)
{evidence_text}

## Task
1. Identify potential drug targets mentioned in the evidence
2. Propose mechanism hypotheses (Drug → Target → Pathway → Effect)
3. Rate confidence based on evidence strength
4. Suggest searches to test each hypothesis

Generate 2-4 hypotheses, prioritized by confidence."""
```

### 4.2 Hypothesis Agent (`src/agents/hypothesis_agent.py`)

```python
"""Hypothesis agent for mechanistic reasoning."""
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)
from pydantic_ai import Agent

from src.prompts.hypothesis import SYSTEM_PROMPT, format_hypothesis_prompt
from src.utils.config import settings
from src.utils.models import Evidence, HypothesisAssessment

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class HypothesisAgent(BaseAgent):
    """Generates mechanistic hypotheses based on evidence."""

    def __init__(
        self,
        evidence_store: dict[str, list[Evidence]],
        embedding_service: "EmbeddingService | None" = None,  # NEW: for diverse selection
    ) -> None:
        super().__init__(
            name="HypothesisAgent",
            description="Generates scientific hypotheses about drug mechanisms to guide research",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service  # Used for MMR evidence selection
        self._agent = Agent(
            model=settings.llm_provider,  # Uses configured LLM
            output_type=HypothesisAssessment,
            system_prompt=SYSTEM_PROMPT,
        )

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Generate hypotheses based on current evidence."""
        # Extract query
        query = self._extract_query(messages)

        # Get current evidence
        evidence = self._evidence_store.get("current", [])

        if not evidence:
            return AgentRunResponse(
                messages=[ChatMessage(
                    role=Role.ASSISTANT,
                    text="No evidence available yet. Search for evidence first."
                )],
                response_id="hypothesis-no-evidence",
            )

        # Generate hypotheses with diverse evidence selection
        # NOTE: format_hypothesis_prompt is now async
        prompt = await format_hypothesis_prompt(
            query, evidence, embeddings=self._embeddings
        )
        result = await self._agent.run(prompt)
        assessment = result.output

        # Store hypotheses in shared context
        existing = self._evidence_store.get("hypotheses", [])
        self._evidence_store["hypotheses"] = existing + assessment.hypotheses

        # Format response
        response_text = self._format_response(assessment)

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"hypothesis-{len(assessment.hypotheses)}",
            additional_properties={"assessment": assessment.model_dump()},
        )

    def _format_response(self, assessment: HypothesisAssessment) -> str:
        """Format hypothesis assessment as markdown."""
        lines = ["## Generated Hypotheses\n"]

        for i, h in enumerate(assessment.hypotheses, 1):
            lines.append(f"### Hypothesis {i} (Confidence: {h.confidence:.0%})")
            lines.append(f"**Mechanism**: {h.drug} → {h.target} → {h.pathway} → {h.effect}")
            lines.append(f"**Suggested searches**: {', '.join(h.search_suggestions)}\n")

        if assessment.primary_hypothesis:
            lines.append(f"### Primary Hypothesis")
            h = assessment.primary_hypothesis
            lines.append(f"{h.drug} → {h.target} → {h.pathway} → {h.effect}\n")

        if assessment.knowledge_gaps:
            lines.append("### Knowledge Gaps")
            for gap in assessment.knowledge_gaps:
                lines.append(f"- {gap}")

        if assessment.recommended_searches:
            lines.append("\n### Recommended Next Searches")
            for search in assessment.recommended_searches:
                lines.append(f"- `{search}`")

        return "\n".join(lines)

    def _extract_query(self, messages) -> str:
        """Extract query from messages."""
        if isinstance(messages, str):
            return messages
        elif isinstance(messages, ChatMessage):
            return messages.text or ""
        elif isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER:
                    return msg.text or ""
                elif isinstance(msg, str):
                    return msg
        return ""

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper."""
        result = await self.run(messages, thread=thread, **kwargs)
        yield AgentRunResponseUpdate(
            messages=result.messages,
            response_id=result.response_id
        )
```

### 4.3 Update MagenticOrchestrator

Add HypothesisAgent to the workflow:

```python
# In MagenticOrchestrator.__init__
self._hypothesis_agent = HypothesisAgent(self._evidence_store)

# In workflow building
workflow = (
    MagenticBuilder()
    .participants(
        searcher=search_agent,
        hypothesizer=self._hypothesis_agent,  # NEW
        judge=judge_agent,
    )
    .with_standard_manager(...)
    .build()
)

# Update task instruction
task = f"""Research drug repurposing opportunities for: {query}

Workflow:
1. SearchAgent: Find initial evidence from PubMed and web
2. HypothesisAgent: Generate mechanistic hypotheses (Drug → Target → Pathway → Effect)
3. SearchAgent: Use hypothesis-suggested queries for targeted search
4. JudgeAgent: Evaluate if evidence supports hypotheses
5. Repeat until confident or max rounds

Focus on:
- Identifying specific molecular targets
- Understanding mechanism of action
- Finding supporting/contradicting evidence for hypotheses
"""
```

---

## 5. Directory Structure After Phase 7

```
src/
├── agents/
│   ├── search_agent.py
│   ├── judge_agent.py
│   └── hypothesis_agent.py     # NEW
├── prompts/
│   ├── judge.py
│   └── hypothesis.py           # NEW
├── services/
│   └── embeddings.py
└── utils/
    └── models.py               # Updated with hypothesis models
```

---

## 6. Tests

### 6.1 Unit Tests (`tests/unit/agents/test_hypothesis_agent.py`)

```python
"""Unit tests for HypothesisAgent."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.hypothesis_agent import HypothesisAgent
from src.utils.models import Citation, Evidence, HypothesisAssessment, MechanismHypothesis


@pytest.fixture
def sample_evidence():
    return [
        Evidence(
            content="Metformin activates AMPK, which inhibits mTOR signaling...",
            citation=Citation(
                source="pubmed",
                title="Metformin and AMPK",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023"
            )
        )
    ]


@pytest.fixture
def mock_assessment():
    return HypothesisAssessment(
        hypotheses=[
            MechanismHypothesis(
                drug="Metformin",
                target="AMPK",
                pathway="mTOR inhibition",
                effect="Reduced cancer cell proliferation",
                confidence=0.75,
                search_suggestions=["metformin AMPK cancer", "mTOR cancer therapy"]
            )
        ],
        primary_hypothesis=None,
        knowledge_gaps=["Clinical trial data needed"],
        recommended_searches=["metformin clinical trial cancer"]
    )


@pytest.mark.asyncio
async def test_hypothesis_agent_generates_hypotheses(sample_evidence, mock_assessment):
    """HypothesisAgent should generate mechanistic hypotheses."""
    store = {"current": sample_evidence, "hypotheses": []}

    with patch("src.agents.hypothesis_agent.Agent") as MockAgent:
        mock_result = MagicMock()
        mock_result.output = mock_assessment
        MockAgent.return_value.run = AsyncMock(return_value=mock_result)

        agent = HypothesisAgent(store)
        response = await agent.run("metformin cancer")

        assert "AMPK" in response.messages[0].text
        assert len(store["hypotheses"]) == 1


@pytest.mark.asyncio
async def test_hypothesis_agent_no_evidence():
    """HypothesisAgent should handle empty evidence gracefully."""
    store = {"current": [], "hypotheses": []}
    agent = HypothesisAgent(store)

    response = await agent.run("test query")

    assert "No evidence" in response.messages[0].text
```

---

## 7. Definition of Done

Phase 7 is **COMPLETE** when:

1. `MechanismHypothesis` and `HypothesisAssessment` models implemented
2. `HypothesisAgent` generates hypotheses from evidence
3. Hypotheses stored in shared context
4. Search queries generated from hypotheses
5. Magentic workflow includes HypothesisAgent
6. All unit tests pass

---

## 8. Value Delivered

| Before (Phase 6) | After (Phase 7) |
|------------------|-----------------|
| Reactive search | Hypothesis-driven search |
| Generic queries | Mechanism-targeted queries |
| No scientific reasoning | Drug → Target → Pathway → Effect |
| Judge says "need more" | Hypothesis says "search for X to test Y" |

**Real example improvement:**
- Query: "metformin alzheimer"
- Before: "metformin alzheimer mechanism", "metformin brain"
- After: "metformin AMPK activation", "AMPK autophagy neurodegeneration", "autophagy amyloid clearance"

The search becomes **scientifically targeted** rather than keyword variations.
