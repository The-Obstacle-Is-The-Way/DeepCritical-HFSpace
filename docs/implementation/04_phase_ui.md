# Phase 4 Implementation Spec: Orchestrator & UI

**Goal**: Connect the Brain and the Body, then give it a Face.
**Philosophy**: "Streaming is Trust."
**Prerequisite**: Phase 3 complete (all judge tests passing)

---

## 1. The Slice Definition

This slice connects:
1. **Orchestrator**: The state machine (While loop) calling Search -> Judge.
2. **UI**: Gradio interface that visualizes the loop.

**Files to Create/Modify**:
- `src/orchestrator.py` - Agent loop logic
- `src/app.py` - Gradio UI
- `tests/unit/test_orchestrator.py` - Unit tests
- `Dockerfile` - Container for deployment
- `README.md` - Usage instructions (update)

---

## 2. Agent Events (`src/utils/models.py`)

Add event types for streaming UI updates:

```python
"""Add to src/utils/models.py (after JudgeAssessment models)."""
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime


class AgentEvent(BaseModel):
    """Event emitted by the orchestrator for UI streaming."""

    type: Literal[
        "started",
        "searching",
        "search_complete",
        "judging",
        "judge_complete",
        "looping",
        "synthesizing",
        "complete",
        "error",
    ]
    message: str
    data: Any = None
    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int = 0

    def to_markdown(self) -> str:
        """Format event as markdown for chat display."""
        icons = {
            "started": "🚀",
            "searching": "🔍",
            "search_complete": "📚",
            "judging": "🧠",
            "judge_complete": "✅",
            "looping": "🔄",
            "synthesizing": "📝",
            "complete": "🎉",
            "error": "❌",
        }
        icon = icons.get(self.type, "•")
        return f"{icon} **{self.type.upper()}**: {self.message}"


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_iterations: int = Field(default=5, ge=1, le=10)
    max_results_per_tool: int = Field(default=10, ge=1, le=50)
    search_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
```

---

## 3. The Orchestrator (`src/orchestrator.py`)

This is the "Agent" logic — the while loop that drives search and judgment.

```python
"""Orchestrator - the agent loop connecting Search and Judge."""
import asyncio
from typing import AsyncGenerator, List, Protocol
import structlog

from src.utils.models import (
    Evidence,
    SearchResult,
    JudgeAssessment,
    AgentEvent,
    OrchestratorConfig,
)

logger = structlog.get_logger()


class SearchHandlerProtocol(Protocol):
    """Protocol for search handler."""
    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        ...


class JudgeHandlerProtocol(Protocol):
    """Protocol for judge handler."""
    async def assess(self, question: str, evidence: List[Evidence]) -> JudgeAssessment:
        ...


class Orchestrator:
    """
    The agent orchestrator - runs the Search -> Judge -> Loop cycle.

    This is a generator-based design that yields events for real-time UI updates.
    """

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        judge_handler: JudgeHandlerProtocol,
        config: OrchestratorConfig | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            search_handler: Handler for executing searches
            judge_handler: Handler for assessing evidence
            config: Optional configuration (uses defaults if not provided)
        """
        self.search = search_handler
        self.judge = judge_handler
        self.config = config or OrchestratorConfig()
        self.history: List[dict] = []

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the agent loop for a query.

        Yields AgentEvent objects for each step, allowing real-time UI updates.

        Args:
            query: The user's research question

        Yields:
            AgentEvent objects for each step of the process
        """
        logger.info("Starting orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research for: {query}",
            iteration=0,
        )

        all_evidence: List[Evidence] = []
        current_queries = [query]
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info("Iteration", iteration=iteration, queries=current_queries)

            # === SEARCH PHASE ===
            yield AgentEvent(
                type="searching",
                message=f"Searching for: {', '.join(current_queries[:3])}...",
                iteration=iteration,
            )

            try:
                # Execute searches for all current queries
                search_tasks = [
                    self.search.execute(q, self.config.max_results_per_tool)
                    for q in current_queries[:3]  # Limit to 3 queries per iteration
                ]
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Collect evidence from successful searches
                new_evidence: List[Evidence] = []
                errors: List[str] = []

                for q, result in zip(current_queries[:3], search_results):
                    if isinstance(result, Exception):
                        errors.append(f"Search for '{q}' failed: {str(result)}")
                    else:
                        new_evidence.extend(result.evidence)
                        errors.extend(result.errors)

                # Deduplicate evidence by URL
                seen_urls = {e.citation.url for e in all_evidence}
                unique_new = [e for e in new_evidence if e.citation.url not in seen_urls]
                all_evidence.extend(unique_new)

                yield AgentEvent(
                    type="search_complete",
                    message=f"Found {len(unique_new)} new sources ({len(all_evidence)} total)",
                    data={"new_count": len(unique_new), "total_count": len(all_evidence)},
                    iteration=iteration,
                )

                if errors:
                    logger.warning("Search errors", errors=errors)

            except Exception as e:
                logger.error("Search phase failed", error=str(e))
                yield AgentEvent(
                    type="error",
                    message=f"Search failed: {str(e)}",
                    iteration=iteration,
                )
                continue

            # === JUDGE PHASE ===
            yield AgentEvent(
                type="judging",
                message=f"Evaluating {len(all_evidence)} sources...",
                iteration=iteration,
            )

            try:
                assessment = await self.judge.assess(query, all_evidence)

                yield AgentEvent(
                    type="judge_complete",
                    message=f"Assessment: {assessment.recommendation} (confidence: {assessment.confidence:.0%})",
                    data={
                        "sufficient": assessment.sufficient,
                        "confidence": assessment.confidence,
                        "mechanism_score": assessment.details.mechanism_score,
                        "clinical_score": assessment.details.clinical_evidence_score,
                    },
                    iteration=iteration,
                )

                # Record this iteration in history
                self.history.append({
                    "iteration": iteration,
                    "queries": current_queries,
                    "evidence_count": len(all_evidence),
                    "assessment": assessment.model_dump(),
                })

                # === DECISION PHASE ===
                if assessment.sufficient and assessment.recommendation == "synthesize":
                    yield AgentEvent(
                        type="synthesizing",
                        message="Evidence sufficient! Preparing synthesis...",
                        iteration=iteration,
                    )

                    # Generate final response
                    final_response = self._generate_synthesis(query, all_evidence, assessment)

                    yield AgentEvent(
                        type="complete",
                        message=final_response,
                        data={
                            "evidence_count": len(all_evidence),
                            "iterations": iteration,
                            "drug_candidates": assessment.details.drug_candidates,
                            "key_findings": assessment.details.key_findings,
                        },
                        iteration=iteration,
                    )
                    return

                else:
                    # Need more evidence - prepare next queries
                    current_queries = assessment.next_search_queries or [
                        f"{query} mechanism of action",
                        f"{query} clinical evidence",
                    ]

                    yield AgentEvent(
                        type="looping",
                        message=f"Need more evidence. Next searches: {', '.join(current_queries[:2])}...",
                        data={"next_queries": current_queries},
                        iteration=iteration,
                    )

            except Exception as e:
                logger.error("Judge phase failed", error=str(e))
                yield AgentEvent(
                    type="error",
                    message=f"Assessment failed: {str(e)}",
                    iteration=iteration,
                )
                continue

        # Max iterations reached
        yield AgentEvent(
            type="complete",
            message=self._generate_partial_synthesis(query, all_evidence),
            data={
                "evidence_count": len(all_evidence),
                "iterations": iteration,
                "max_reached": True,
            },
            iteration=iteration,
        )

    def _generate_synthesis(
        self,
        query: str,
        evidence: List[Evidence],
        assessment: JudgeAssessment,
    ) -> str:
        """
        Generate the final synthesis response.

        Args:
            query: The original question
            evidence: All collected evidence
            assessment: The final assessment

        Returns:
            Formatted synthesis as markdown
        """
        drug_list = "\n".join([f"- **{d}**" for d in assessment.details.drug_candidates]) or "- No specific candidates identified"
        findings_list = "\n".join([f"- {f}" for f in assessment.details.key_findings]) or "- See evidence below"

        citations = "\n".join([
            f"{i+1}. [{e.citation.title}]({e.citation.url}) ({e.citation.source.upper()}, {e.citation.date})"
            for i, e in enumerate(evidence[:10])  # Limit to 10 citations
        ])

        return f"""## Drug Repurposing Analysis

### Question
{query}

### Drug Candidates
{drug_list}

### Key Findings
{findings_list}

### Assessment
- **Mechanism Score**: {assessment.details.mechanism_score}/10
- **Clinical Evidence Score**: {assessment.details.clinical_evidence_score}/10
- **Confidence**: {assessment.confidence:.0%}

### Reasoning
{assessment.reasoning}

### Citations ({len(evidence)} sources)
{citations}

---
*Analysis based on {len(evidence)} sources across {len(self.history)} iterations.*
"""

    def _generate_partial_synthesis(
        self,
        query: str,
        evidence: List[Evidence],
    ) -> str:
        """
        Generate a partial synthesis when max iterations reached.

        Args:
            query: The original question
            evidence: All collected evidence

        Returns:
            Formatted partial synthesis as markdown
        """
        citations = "\n".join([
            f"{i+1}. [{e.citation.title}]({e.citation.url}) ({e.citation.source.upper()})"
            for i, e in enumerate(evidence[:10])
        ])

        return f"""## Partial Analysis (Max Iterations Reached)

### Question
{query}

### Status
Maximum search iterations reached. The evidence gathered may be incomplete.

### Evidence Collected
Found {len(evidence)} sources. Consider refining your query for more specific results.

### Citations
{citations}

---
*Consider searching with more specific terms or drug names.*
"""
```

---

## 4. The Gradio UI (`src/app.py`)

Using Gradio 5 generator pattern for real-time streaming.

```python
"""Gradio UI for DeepBoner agent."""
import asyncio
import gradio as gr
from typing import AsyncGenerator

from src.orchestrator import Orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.biorxiv import BioRxivTool
from src.tools.search_handler import SearchHandler
from src.agent_factory.judges import JudgeHandler, HFInferenceJudgeHandler
from src.utils.models import OrchestratorConfig, AgentEvent


def create_orchestrator(
    user_api_key: str | None = None,
    api_provider: str = "openai",
) -> tuple[Orchestrator, str]:
    """
    Create an orchestrator instance.

    Args:
        user_api_key: Optional user-provided API key (BYOK)
        api_provider: API provider ("openai" or "anthropic")

    Returns:
        Tuple of (Configured Orchestrator instance, backend_name)

    Priority:
        1. User-provided API key → JudgeHandler (OpenAI/Anthropic)
        2. Environment API key → JudgeHandler (OpenAI/Anthropic)
        3. No key → HFInferenceJudgeHandler (FREE, automatic fallback chain)

    HF Inference Fallback Chain:
        1. Llama 3.1 8B (requires HF_TOKEN for gated model)
        2. Mistral 7B (may require token)
        3. Zephyr 7B (ungated, always works)
    """
    import os

    # Create search tools
    search_handler = SearchHandler(
        tools=[PubMedTool(), ClinicalTrialsTool(), BioRxivTool()],
        timeout=30.0,
    )

    # Determine which judge to use
    has_env_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    has_user_key = bool(user_api_key)
    has_hf_token = bool(os.getenv("HF_TOKEN"))

    if has_user_key:
        # User provided their own key
        judge_handler = JudgeHandler(model=None)
        backend_name = f"your {api_provider.upper()} API key"
    elif has_env_key:
        # Environment has API key configured
        judge_handler = JudgeHandler(model=None)
        backend_name = "configured API key"
    else:
        # Use FREE HuggingFace Inference with automatic fallback
        judge_handler = HFInferenceJudgeHandler()
        if has_hf_token:
            backend_name = "HuggingFace Inference (Llama 3.1)"
        else:
            backend_name = "HuggingFace Inference (free tier)"

    # Create orchestrator
    config = OrchestratorConfig(
        max_iterations=5,
        max_results_per_tool=10,
    )

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    ), backend_name


async def research_agent(
    message: str,
    history: list[dict],
    api_key: str = "",
    api_provider: str = "openai",
) -> AsyncGenerator[str, None]:
    """
    Gradio chat function that runs the research agent.

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        api_key: Optional user-provided API key (BYOK)
        api_provider: API provider ("openai" or "anthropic")

    Yields:
        Markdown-formatted responses for streaming
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    import os

    # Clean user-provided API key
    user_api_key = api_key.strip() if api_key else None

    # Create orchestrator with appropriate judge
    orchestrator, backend_name = create_orchestrator(
        user_api_key=user_api_key,
        api_provider=api_provider,
    )

    # Determine icon based on backend
    has_hf_token = bool(os.getenv("HF_TOKEN"))
    if "HuggingFace" in backend_name:
        icon = "🤗"
        extra_note = (
            "\n*For premium analysis, enter an OpenAI or Anthropic API key.*"
            if not has_hf_token else ""
        )
    else:
        icon = "🔑"
        extra_note = ""

    # Inform user which backend is being used
    yield f"{icon} **Using {backend_name}**{extra_note}\n\n"

    # Run the agent and stream events
    response_parts = []

    try:
        async for event in orchestrator.run(message):
            # Format event as markdown
            event_md = event.to_markdown()
            response_parts.append(event_md)

            # If complete, show full response
            if event.type == "complete":
                yield event.message
            else:
                # Show progress
                yield "\n\n".join(response_parts)

    except Exception as e:
        yield f"❌ **Error**: {str(e)}"


def create_demo() -> gr.Blocks:
    """
    Create the Gradio demo interface.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(
        title="DeepBoner - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🧬 DeepBoner
        ## AI-Powered Drug Repurposing Research Agent

        Ask questions about potential drug repurposing opportunities.
        The agent will search PubMed and the web, evaluate evidence, and provide recommendations.

        **Example questions:**
        - "What drugs could be repurposed for Alzheimer's disease?"
        - "Is metformin effective for cancer treatment?"
        - "What existing medications show promise for Long COVID?"
        """)

        # Note: additional_inputs render in an accordion below the chat input
        gr.ChatInterface(
            fn=research_agent,
            examples=[
                [
                    "What drugs could be repurposed for Alzheimer's disease?",
                    "simple",
                    "",
                    "openai",
                ],
                [
                    "Is metformin effective for treating cancer?",
                    "simple",
                    "",
                    "openai",
                ],
            ],
            additional_inputs=[
                gr.Radio(
                    choices=["simple", "magentic"],
                    value="simple",
                    label="Orchestrator Mode",
                    info="Simple: Linear | Magentic: Multi-Agent (OpenAI)",
                ),
                gr.Textbox(
                    label="API Key (Optional - Bring Your Own Key)",
                    placeholder="sk-... or sk-ant-...",
                    type="password",
                    info="Enter your own API key for full AI analysis. Never stored.",
                ),
                gr.Radio(
                    choices=["openai", "anthropic"],
                    value="openai",
                    label="API Provider",
                    info="Select the provider for your API key",
                ),
            ],
        )

        gr.Markdown("""
        ---
        **Note**: This is a research tool and should not be used for medical decisions.
        Always consult healthcare professionals for medical advice.

        Built with 🤖 PydanticAI + 🔬 PubMed + 🦆 DuckDuckGo
        """)

    return demo


def main():
    """Run the Gradio app."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
```

---

## 5. TDD Workflow

### Test File: `tests/unit/test_orchestrator.py`

```python
"""Unit tests for Orchestrator."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.utils.models import (
    Evidence,
    Citation,
    SearchResult,
    JudgeAssessment,
    AssessmentDetails,
    OrchestratorConfig,
)


class TestOrchestrator:
    """Tests for Orchestrator."""

    @pytest.fixture
    def mock_search_handler(self):
        """Create a mock search handler."""
        handler = AsyncMock()
        handler.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[
                Evidence(
                    content="Test content",
                    citation=Citation(
                        source="pubmed",
                        title="Test Title",
                        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                        date="2024-01-01",
                    ),
                ),
            ],
            sources_searched=["pubmed"],
            total_found=1,
            errors=[],
        ))
        return handler

    @pytest.fixture
    def mock_judge_sufficient(self):
        """Create a mock judge that returns sufficient."""
        handler = AsyncMock()
        handler.assess = AsyncMock(return_value=JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=8,
                mechanism_reasoning="Good mechanism",
                clinical_evidence_score=7,
                clinical_reasoning="Good clinical",
                drug_candidates=["Drug A"],
                key_findings=["Finding 1"],
            ),
            sufficient=True,
            confidence=0.85,
            recommendation="synthesize",
            next_search_queries=[],
            reasoning="Evidence is sufficient",
        ))
        return handler

    @pytest.fixture
    def mock_judge_insufficient(self):
        """Create a mock judge that returns insufficient."""
        handler = AsyncMock()
        handler.assess = AsyncMock(return_value=JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=4,
                mechanism_reasoning="Weak mechanism",
                clinical_evidence_score=3,
                clinical_reasoning="Weak clinical",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.3,
            recommendation="continue",
            next_search_queries=["more specific query"],
            reasoning="Need more evidence",
        ))
        return handler

    @pytest.mark.asyncio
    async def test_orchestrator_completes_with_sufficient_evidence(
        self,
        mock_search_handler,
        mock_judge_sufficient,
    ):
        """Orchestrator should complete when evidence is sufficient."""
        from src.orchestrator import Orchestrator

        config = OrchestratorConfig(max_iterations=5)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_sufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have started, searched, judged, and completed
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "searching" in event_types
        assert "search_complete" in event_types
        assert "judging" in event_types
        assert "judge_complete" in event_types
        assert "complete" in event_types

        # Should only have 1 iteration
        complete_event = [e for e in events if e.type == "complete"][0]
        assert complete_event.iteration == 1

    @pytest.mark.asyncio
    async def test_orchestrator_loops_when_insufficient(
        self,
        mock_search_handler,
        mock_judge_insufficient,
    ):
        """Orchestrator should loop when evidence is insufficient."""
        from src.orchestrator import Orchestrator

        config = OrchestratorConfig(max_iterations=3)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have looping events
        event_types = [e.type for e in events]
        assert event_types.count("looping") >= 2  # At least 2 loop events

        # Should hit max iterations
        complete_event = [e for e in events if e.type == "complete"][0]
        assert complete_event.data.get("max_reached") is True

    @pytest.mark.asyncio
    async def test_orchestrator_respects_max_iterations(
        self,
        mock_search_handler,
        mock_judge_insufficient,
    ):
        """Orchestrator should stop at max_iterations."""
        from src.orchestrator import Orchestrator

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have exactly 2 iterations
        max_iteration = max(e.iteration for e in events)
        assert max_iteration == 2

    @pytest.mark.asyncio
    async def test_orchestrator_handles_search_error(self):
        """Orchestrator should handle search errors gracefully."""
        from src.orchestrator import Orchestrator

        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(side_effect=Exception("Search failed"))

        mock_judge = AsyncMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            details=AssessmentDetails(
                mechanism_score=0,
                mechanism_reasoning="N/A",
                clinical_evidence_score=0,
                clinical_reasoning="N/A",
                drug_candidates=[],
                key_findings=[],
            ),
            sufficient=False,
            confidence=0.0,
            recommendation="continue",
            next_search_queries=["retry query"],
            reasoning="Search failed",
        ))

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have error events
        event_types = [e.type for e in events]
        assert "error" in event_types

    @pytest.mark.asyncio
    async def test_orchestrator_deduplicates_evidence(self, mock_judge_insufficient):
        """Orchestrator should deduplicate evidence by URL."""
        from src.orchestrator import Orchestrator

        # Search returns same evidence each time
        duplicate_evidence = Evidence(
            content="Duplicate content",
            citation=Citation(
                source="pubmed",
                title="Same Title",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",  # Same URL
                date="2024-01-01",
            ),
        )

        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[duplicate_evidence],
            sources_searched=["pubmed"],
            total_found=1,
            errors=[],
        ))

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Second search_complete should show 0 new evidence
        search_complete_events = [e for e in events if e.type == "search_complete"]
        assert len(search_complete_events) == 2

        # First iteration should have 1 new
        assert search_complete_events[0].data["new_count"] == 1

        # Second iteration should have 0 new (duplicate)
        assert search_complete_events[1].data["new_count"] == 0


class TestAgentEvent:
    """Tests for AgentEvent."""

    def test_to_markdown(self):
        """AgentEvent should format to markdown correctly."""
        from src.utils.models import AgentEvent

        event = AgentEvent(
            type="searching",
            message="Searching for: metformin alzheimer",
            iteration=1,
        )

        md = event.to_markdown()
        assert "🔍" in md
        assert "SEARCHING" in md
        assert "metformin alzheimer" in md

    def test_complete_event_icon(self):
        """Complete event should have celebration icon."""
        from src.utils.models import AgentEvent

        event = AgentEvent(
            type="complete",
            message="Done!",
            iteration=3,
        )

        md = event.to_markdown()
        assert "🎉" in md
```

---

## 6. Dockerfile

```dockerfile
# Dockerfile for DeepBoner
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install --system .

# Expose port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the app
CMD ["python", "-m", "src.app"]
```

---

## 7. HuggingFace Spaces Configuration

Create `README.md` header for HuggingFace Spaces:

```markdown
---
title: DeepBoner
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: src/app.py
pinned: false
license: mit
---

# DeepBoner

AI-Powered Drug Repurposing Research Agent
```

---

## 8. Implementation Checklist

- [ ] Add `AgentEvent` and `OrchestratorConfig` models to `src/utils/models.py`
- [ ] Implement `src/orchestrator.py` with full Orchestrator class
- [ ] Implement `src/app.py` with Gradio interface
- [ ] Create `tests/unit/test_orchestrator.py` with all tests
- [ ] Create `Dockerfile` for deployment
- [ ] Update project `README.md` with usage instructions
- [ ] Run `uv run pytest tests/unit/test_orchestrator.py -v` — **ALL TESTS MUST PASS**
- [ ] Test locally: `uv run python -m src.app`
- [ ] Commit: `git commit -m "feat: phase 4 orchestrator and UI complete"`

---

## 9. Definition of Done

Phase 4 is **COMPLETE** when:

1. All unit tests pass: `uv run pytest tests/unit/test_orchestrator.py -v`
2. Orchestrator correctly loops Search -> Judge until sufficient
3. Max iterations limit is enforced
4. Graceful error handling throughout
5. Gradio UI streams events in real-time
6. Can run locally:

```bash
# Start the UI
uv run python -m src.app

# Open browser to http://localhost:7860
# Enter a question like "What drugs could be repurposed for Alzheimer's disease?"
# Watch the agent search, evaluate, and respond
```

7. Can run the full flow in Python:

```python
import asyncio
from src.orchestrator import Orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.biorxiv import BioRxivTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.search_handler import SearchHandler
from src.agent_factory.judges import HFInferenceJudgeHandler, MockJudgeHandler
from src.utils.models import OrchestratorConfig

async def test_full_flow():
    # Create components
    search_handler = SearchHandler([PubMedTool(), ClinicalTrialsTool(), BioRxivTool()])

    # Option 1: Use FREE HuggingFace Inference (real AI analysis)
    judge_handler = HFInferenceJudgeHandler()

    # Option 2: Use MockJudgeHandler for UNIT TESTING ONLY
    # judge_handler = MockJudgeHandler()

    config = OrchestratorConfig(max_iterations=3)

    # Create orchestrator
    orchestrator = Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )

    # Run and collect events
    print("Starting agent...")
    async for event in orchestrator.run("metformin alzheimer"):
        print(event.to_markdown())

    print("\nDone!")

asyncio.run(test_full_flow())
```

**Important**: `MockJudgeHandler` is for **unit testing only**. For actual demo/production use, always use `HFInferenceJudgeHandler` (free) or `JudgeHandler` (with API key).

---

## 10. Deployment Verification

After deployment to HuggingFace Spaces:

1. **Visit the Space URL** and verify the UI loads
2. **Test with example queries**:
   - "What drugs could be repurposed for Alzheimer's disease?"
   - "Is metformin effective for cancer treatment?"
3. **Verify streaming** - events should appear in real-time
4. **Check error handling** - try an empty query, verify graceful handling
5. **Monitor logs** for any errors

---

## Project Complete! 🎉

When Phase 4 is done, the DeepBoner MVP is complete:

- **Phase 1**: Foundation (uv, pytest, config) ✅
- **Phase 2**: Search Slice (PubMed, DuckDuckGo) ✅
- **Phase 3**: Judge Slice (PydanticAI, structured output) ✅
- **Phase 4**: Orchestrator + UI (Gradio, streaming) ✅

The agent can:
1. Accept a drug repurposing question
2. Search PubMed and the web for evidence
3. Evaluate evidence quality with an LLM
4. Loop until confident or max iterations
5. Synthesize a research-backed recommendation
6. Display real-time progress in a beautiful UI
