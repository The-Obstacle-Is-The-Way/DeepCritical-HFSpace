# Phase 4 Implementation Spec: Orchestrator & UI

**Goal**: Connect the Brain and the Body, then give it a Face.
**Philosophy**: "Streaming is Trust."
**Estimated Effort**: 4-5 hours
**Prerequisite**: Phases 1-3 complete (Search + Judge slices working)

---

## 1. The Slice Definition

This slice connects everything:
1. **Orchestrator**: The state machine (while loop) calling Search → Judge → (loop or synthesize).
2. **UI**: Gradio 5 interface with real-time streaming events.
3. **Deployment**: HuggingFace Spaces configuration.

**Directories**:
- `src/features/orchestrator/`
- `src/app.py`

---

## 2. Models (`src/features/orchestrator/models.py`)

```python
"""Data models for the Orchestrator feature."""
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime
from enum import Enum


class AgentState(str, Enum):
    """Possible states of the agent."""
    IDLE = "idle"
    SEARCHING = "searching"
    JUDGING = "judging"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


class AgentEvent(BaseModel):
    """An event emitted by the agent during execution."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state: AgentState
    message: str
    iteration: int = 0
    data: dict[str, Any] | None = None

    def to_display(self) -> str:
        """Format for UI display."""
        emoji_map = {
            AgentState.SEARCHING: "🔍",
            AgentState.JUDGING: "🧠",
            AgentState.SYNTHESIZING: "📝",
            AgentState.COMPLETE: "✅",
            AgentState.ERROR: "❌",
            AgentState.IDLE: "⏸️",
        }
        emoji = emoji_map.get(self.state, "")
        return f"{emoji} **[{self.state.value.upper()}]** {self.message}"


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_iterations: int = Field(default=10, ge=1, le=50)
    max_evidence_per_iteration: int = Field(default=10, ge=1, le=50)
    search_timeout: float = Field(default=30.0, description="Seconds")

    # Budget constraints
    max_llm_calls: int = Field(default=20, description="Max LLM API calls")

    # Quality thresholds
    min_quality_score: int = Field(default=6, ge=0, le=10)


class SessionState(BaseModel):
    """State of an orchestrator session."""

    session_id: str
    question: str
    iterations_completed: int = 0
    total_evidence: int = 0
    llm_calls: int = 0
    current_state: AgentState = AgentState.IDLE
    final_report: str | None = None
    error: str | None = None
```

---

## 3. Orchestrator (`src/features/orchestrator/handlers.py`)

The core agent loop.

```python
"""Orchestrator - the main agent loop."""
import asyncio
from typing import AsyncGenerator
import structlog

from src.shared.config import settings
from src.shared.exceptions import DeepCriticalError
from src.features.search.handlers import SearchHandler
from src.features.search.tools import PubMedTool, WebTool
from src.features.search.models import Evidence
from src.features.judge.handlers import JudgeHandler
from src.features.judge.models import JudgeAssessment
from .models import AgentEvent, AgentState, OrchestratorConfig, SessionState

logger = structlog.get_logger()


class Orchestrator:
    """Main agent orchestrator - coordinates search, judge, and synthesis."""

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        search_handler: SearchHandler | None = None,
        judge_handler: JudgeHandler | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            search_handler: Injected search handler (for testing)
            judge_handler: Injected judge handler (for testing)
        """
        self.config = config or OrchestratorConfig(
            max_iterations=settings.max_iterations,
        )

        # Initialize handlers (or use injected ones for testing)
        self.search = search_handler or SearchHandler(
            tools=[PubMedTool(), WebTool()],
            timeout=self.config.search_timeout,
        )
        self.judge = judge_handler or JudgeHandler()

    async def run(
        self,
        question: str,
        session_id: str = "default",
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the agent loop, yielding events for the UI.

        This is an async generator that yields AgentEvent objects
        as the agent progresses through its workflow.

        Args:
            question: The research question to answer
            session_id: Unique session identifier

        Yields:
            AgentEvent objects describing the agent's progress
        """
        logger.info("Starting orchestrator run", question=question[:100])

        # Initialize state
        state = SessionState(
            session_id=session_id,
            question=question,
        )
        all_evidence: list[Evidence] = []
        current_queries = [question]  # Start with the original question

        try:
            # Main agent loop
            while state.iterations_completed < self.config.max_iterations:
                state.iterations_completed += 1
                iteration = state.iterations_completed

                # --- SEARCH PHASE ---
                state.current_state = AgentState.SEARCHING
                yield AgentEvent(
                    state=AgentState.SEARCHING,
                    message=f"Searching for evidence (iteration {iteration}/{self.config.max_iterations})",
                    iteration=iteration,
                    data={"queries": current_queries},
                )

                # Execute searches for all current queries
                for query in current_queries[:3]:  # Limit to 3 queries per iteration
                    search_result = await self.search.execute(
                        query,
                        max_results_per_tool=self.config.max_evidence_per_iteration,
                    )
                    # Add new evidence (avoid duplicates by URL)
                    existing_urls = {e.citation.url for e in all_evidence}
                    for ev in search_result.evidence:
                        if ev.citation.url not in existing_urls:
                            all_evidence.append(ev)
                            existing_urls.add(ev.citation.url)

                state.total_evidence = len(all_evidence)

                yield AgentEvent(
                    state=AgentState.SEARCHING,
                    message=f"Found {len(all_evidence)} total pieces of evidence",
                    iteration=iteration,
                    data={"total_evidence": len(all_evidence)},
                )

                # --- JUDGE PHASE ---
                state.current_state = AgentState.JUDGING
                yield AgentEvent(
                    state=AgentState.JUDGING,
                    message="Evaluating evidence quality...",
                    iteration=iteration,
                )

                # Check LLM budget
                if state.llm_calls >= self.config.max_llm_calls:
                    yield AgentEvent(
                        state=AgentState.ERROR,
                        message=f"LLM call budget exceeded ({self.config.max_llm_calls} calls)",
                        iteration=iteration,
                    )
                    break

                assessment = await self.judge.assess(question, all_evidence)
                state.llm_calls += 1

                yield AgentEvent(
                    state=AgentState.JUDGING,
                    message=f"Quality: {assessment.overall_quality_score}/10 | "
                            f"Sufficient: {assessment.sufficient}",
                    iteration=iteration,
                    data={
                        "sufficient": assessment.sufficient,
                        "quality_score": assessment.overall_quality_score,
                        "recommendation": assessment.recommendation,
                        "candidates": len(assessment.candidates),
                    },
                )

                # --- DECISION POINT ---
                if assessment.sufficient and assessment.recommendation == "synthesize":
                    # Ready to synthesize!
                    state.current_state = AgentState.SYNTHESIZING
                    yield AgentEvent(
                        state=AgentState.SYNTHESIZING,
                        message="Evidence is sufficient. Generating report...",
                        iteration=iteration,
                    )

                    # Generate the final report
                    report = await self._synthesize_report(
                        question, all_evidence, assessment
                    )
                    state.final_report = report
                    state.llm_calls += 1

                    state.current_state = AgentState.COMPLETE
                    yield AgentEvent(
                        state=AgentState.COMPLETE,
                        message="Research complete!",
                        iteration=iteration,
                        data={
                            "total_iterations": iteration,
                            "total_evidence": len(all_evidence),
                            "llm_calls": state.llm_calls,
                        },
                    )

                    # Yield the final report as a separate event
                    yield AgentEvent(
                        state=AgentState.COMPLETE,
                        message=report,
                        iteration=iteration,
                        data={"is_report": True},
                    )
                    return

                else:
                    # Need more evidence
                    current_queries = assessment.next_search_queries
                    if not current_queries:
                        # No more queries suggested, use gaps as queries
                        current_queries = [f"{question} {gap}" for gap in assessment.gaps[:2]]

                    yield AgentEvent(
                        state=AgentState.JUDGING,
                        message=f"Need more evidence. Next queries: {current_queries[:2]}",
                        iteration=iteration,
                        data={"next_queries": current_queries},
                    )

            # Loop exhausted without sufficient evidence
            state.current_state = AgentState.COMPLETE
            yield AgentEvent(
                state=AgentState.COMPLETE,
                message=f"Max iterations ({self.config.max_iterations}) reached. "
                        "Generating best-effort report...",
                iteration=state.iterations_completed,
            )

            # Generate best-effort report
            report = await self._synthesize_report(
                question, all_evidence, assessment, best_effort=True
            )
            state.final_report = report

            yield AgentEvent(
                state=AgentState.COMPLETE,
                message=report,
                iteration=state.iterations_completed,
                data={"is_report": True, "best_effort": True},
            )

        except DeepCriticalError as e:
            state.current_state = AgentState.ERROR
            state.error = str(e)
            yield AgentEvent(
                state=AgentState.ERROR,
                message=f"Error: {e}",
                iteration=state.iterations_completed,
            )
            logger.error("Orchestrator error", error=str(e))

        except Exception as e:
            state.current_state = AgentState.ERROR
            state.error = str(e)
            yield AgentEvent(
                state=AgentState.ERROR,
                message=f"Unexpected error: {e}",
                iteration=state.iterations_completed,
            )
            logger.exception("Unexpected orchestrator error")

    async def _synthesize_report(
        self,
        question: str,
        evidence: list[Evidence],
        assessment: JudgeAssessment,
        best_effort: bool = False,
    ) -> str:
        """
        Synthesize a research report from the evidence.

        For MVP, we use the Judge's assessment to build a simple report.
        In a full implementation, this would be a separate Report agent.
        """
        # Build citations
        citations = []
        for i, ev in enumerate(evidence, 1):
            citations.append(f"[{i}] {ev.citation.formatted}")

        # Build drug candidates section
        candidates_text = ""
        if assessment.candidates:
            candidates_text = "\n\n## Drug Candidates\n\n"
            for c in assessment.candidates:
                candidates_text += f"### {c.drug_name}\n"
                candidates_text += f"- **Original Indication**: {c.original_indication}\n"
                candidates_text += f"- **Proposed Use**: {c.proposed_indication}\n"
                candidates_text += f"- **Mechanism**: {c.mechanism}\n"
                candidates_text += f"- **Evidence Strength**: {c.evidence_strength}\n\n"

        # Build the report
        quality_note = ""
        if best_effort:
            quality_note = "\n\n> ⚠️ **Note**: This report was generated with limited evidence.\n"

        report = f"""# Drug Repurposing Research Report

## Research Question
{question}
{quality_note}
## Summary
{assessment.reasoning}

**Quality Score**: {assessment.overall_quality_score}/10
**Evidence Coverage**: {assessment.coverage_score}/10
{candidates_text}
## Gaps & Limitations
{chr(10).join(f'- {gap}' for gap in assessment.gaps) if assessment.gaps else '- None identified'}

## References
{chr(10).join(citations[:10])}

---
*Generated by DeepCritical Research Agent*
"""
        return report
```

---

## 4. Gradio UI (`src/app.py`)

```python
"""Gradio UI for DeepCritical Research Agent."""
import gradio as gr
import asyncio
from typing import AsyncGenerator
import uuid

from src.features.orchestrator.handlers import Orchestrator
from src.features.orchestrator.models import AgentState, OrchestratorConfig


# Create a shared orchestrator instance
orchestrator = Orchestrator(
    config=OrchestratorConfig(
        max_iterations=10,
        max_llm_calls=20,
    )
)


async def research_agent(
    message: str,
    history: list[dict],
) -> AsyncGenerator[str, None]:
    """
    Main chat function for Gradio.

    This is an async generator that yields messages as the agent progresses.
    Gradio 5 supports streaming via generators.
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    session_id = str(uuid.uuid4())
    accumulated_output = ""

    async for event in orchestrator.run(message, session_id):
        # Format the event for display
        display = event.to_display()

        # Check if this is the final report
        if event.data and event.data.get("is_report"):
            # Yield the full report
            accumulated_output += f"\n\n{event.message}"
        else:
            accumulated_output += f"\n{display}"

        yield accumulated_output


def create_app() -> gr.Blocks:
    """Create the Gradio app."""

    with gr.Blocks(
        title="DeepCritical - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
# 🔬 DeepCritical Research Agent

AI-powered drug repurposing research assistant. Ask questions about potential
drug repurposing opportunities and get evidence-based answers.

**Example questions:**
- "What existing drugs might help treat long COVID fatigue?"
- "Can metformin be repurposed for Alzheimer's disease?"
- "What is the evidence for statins in cancer treatment?"
        """)

        chatbot = gr.Chatbot(
            label="Research Chat",
            height=500,
            type="messages",  # Use the new messages format
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your Research Question",
                placeholder="Enter your drug repurposing research question...",
                scale=4,
            )
            submit = gr.Button("🔍 Research", variant="primary", scale=1)

        # Clear button
        clear = gr.Button("Clear Chat")

        # Examples
        gr.Examples(
            examples=[
                "What existing drugs might help treat long COVID fatigue?",
                "Can metformin be repurposed for Alzheimer's disease?",
                "What is the evidence for statins in treating cancer?",
                "Are there any approved drugs that could treat ALS?",
            ],
            inputs=msg,
        )

        # Wire up the interface
        async def respond(message, chat_history):
            """Handle user message and stream response."""
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})

            # Stream the response
            response = ""
            async for chunk in research_agent(message, chat_history):
                response = chunk
                yield "", chat_history + [{"role": "assistant", "content": response}]

        submit.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )
        clear.click(lambda: (None, []), outputs=[msg, chatbot])

    return app


# Entry point
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
```

---

## 5. Deployment Configuration

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY .env.example .env

# Install dependencies
RUN uv sync --no-dev

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "python", "src/app.py"]
```

### `README.md` (HuggingFace Spaces)

This goes in the root of your HuggingFace Space.

```markdown
---
title: DeepCritical
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: src/app.py
pinned: false
license: mit
---

# DeepCritical - Drug Repurposing Research Agent

AI-powered research agent for discovering drug repurposing opportunities.

## Features
- 🔍 Search PubMed and web sources
- 🧠 AI-powered evidence assessment
- 📝 Structured research reports
- 💬 Interactive chat interface

## Usage
Enter a research question about drug repurposing, such as:
- "What existing drugs might help treat long COVID fatigue?"
- "Can metformin be repurposed for Alzheimer's disease?"

The agent will search medical literature, assess evidence quality,
and generate a research report with citations.

## API Keys
This space requires an OpenAI API key set as a secret (`OPENAI_API_KEY`).
```

### `.env.example` (Updated)

```bash
# LLM Provider - REQUIRED
# Choose one:
OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Agent Configuration
MAX_ITERATIONS=10

# Logging
LOG_LEVEL=INFO

# Optional: NCBI API key for faster PubMed searches
# NCBI_API_KEY=your-ncbi-key
```

---

## 6. TDD Workflow

### Test File: `tests/unit/features/orchestrator/test_orchestrator.py`

```python
"""Unit tests for the Orchestrator."""
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestOrchestratorModels:
    """Tests for Orchestrator data models."""

    def test_agent_event_display(self):
        """AgentEvent.to_display should format correctly."""
        from src.features.orchestrator.models import AgentEvent, AgentState

        event = AgentEvent(
            state=AgentState.SEARCHING,
            message="Looking for evidence",
            iteration=1,
        )

        display = event.to_display()
        assert "🔍" in display
        assert "SEARCHING" in display
        assert "Looking for evidence" in display

    def test_orchestrator_config_defaults(self):
        """OrchestratorConfig should have sensible defaults."""
        from src.features.orchestrator.models import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.max_iterations == 10
        assert config.max_llm_calls == 20

    def test_orchestrator_config_bounds(self):
        """OrchestratorConfig should enforce bounds."""
        from src.features.orchestrator.models import OrchestratorConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrchestratorConfig(max_iterations=100)  # > 50


class TestOrchestrator:
    """Tests for the Orchestrator."""

    @pytest.mark.asyncio
    async def test_run_yields_events(self, mocker):
        """Orchestrator.run should yield AgentEvents."""
        from src.features.orchestrator.handlers import Orchestrator
        from src.features.orchestrator.models import (
            AgentEvent,
            AgentState,
            OrchestratorConfig,
        )
        from src.features.search.models import Evidence, Citation, SearchResult
        from src.features.judge.models import JudgeAssessment

        # Mock search handler
        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[
                Evidence(
                    content="Test evidence",
                    citation=Citation(
                        source="pubmed",
                        title="Test",
                        url="https://example.com",
                        date="2024",
                    ),
                )
            ],
            sources_searched=["pubmed"],
            total_found=1,
        ))

        # Mock judge handler - returns sufficient on first call
        mock_judge = AsyncMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Good evidence",
            overall_quality_score=8,
            coverage_score=7,
        ))

        config = OrchestratorConfig(max_iterations=3)
        orchestrator = Orchestrator(
            config=config,
            search_handler=mock_search,
            judge_handler=mock_judge,
        )

        events = []
        async for event in orchestrator.run("test question"):
            events.append(event)

        # Should have multiple events
        assert len(events) >= 3

        # Check we got expected state transitions
        states = [e.state for e in events]
        assert AgentState.SEARCHING in states
        assert AgentState.JUDGING in states
        assert AgentState.COMPLETE in states

    @pytest.mark.asyncio
    async def test_run_respects_max_iterations(self, mocker):
        """Orchestrator should stop at max_iterations."""
        from src.features.orchestrator.handlers import Orchestrator
        from src.features.orchestrator.models import OrchestratorConfig
        from src.features.search.models import Evidence, Citation, SearchResult
        from src.features.judge.models import JudgeAssessment

        # Mock search
        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[],
            sources_searched=["pubmed"],
            total_found=0,
        ))

        # Mock judge - always returns insufficient
        mock_judge = AsyncMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Need more",
            overall_quality_score=2,
            coverage_score=1,
            next_search_queries=["more stuff"],
        ))

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            config=config,
            search_handler=mock_search,
            judge_handler=mock_judge,
        )

        events = []
        async for event in orchestrator.run("test"):
            events.append(event)

        # Should stop after max_iterations
        max_iteration = max(e.iteration for e in events)
        assert max_iteration <= 2

    @pytest.mark.asyncio
    async def test_run_handles_search_error(self, mocker):
        """Orchestrator should handle search errors gracefully."""
        from src.features.orchestrator.handlers import Orchestrator
        from src.features.orchestrator.models import AgentState, OrchestratorConfig
        from src.shared.exceptions import SearchError

        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(side_effect=SearchError("API down"))

        mock_judge = AsyncMock()

        orchestrator = Orchestrator(
            config=OrchestratorConfig(max_iterations=1),
            search_handler=mock_search,
            judge_handler=mock_judge,
        )

        events = []
        async for event in orchestrator.run("test"):
            events.append(event)

        # Should have an error event
        error_events = [e for e in events if e.state == AgentState.ERROR]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_run_respects_llm_budget(self, mocker):
        """Orchestrator should stop when LLM budget is exceeded."""
        from src.features.orchestrator.handlers import Orchestrator
        from src.features.orchestrator.models import AgentState, OrchestratorConfig
        from src.features.search.models import SearchResult
        from src.features.judge.models import JudgeAssessment

        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[],
            sources_searched=[],
            total_found=0,
        ))

        # Judge always needs more
        mock_judge = AsyncMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Need more",
            overall_quality_score=2,
            coverage_score=1,
            next_search_queries=["more"],
        ))

        config = OrchestratorConfig(
            max_iterations=100,  # High
            max_llm_calls=2,     # Low - should hit this first
        )
        orchestrator = Orchestrator(
            config=config,
            search_handler=mock_search,
            judge_handler=mock_judge,
        )

        events = []
        async for event in orchestrator.run("test"):
            events.append(event)

        # Should have stopped due to budget
        error_events = [e for e in events if "budget" in e.message.lower()]
        assert len(error_events) >= 1
```

---

## 7. Module Exports (`src/features/orchestrator/__init__.py`)

```python
"""Orchestrator feature - main agent loop."""
from .models import AgentEvent, AgentState, OrchestratorConfig, SessionState
from .handlers import Orchestrator

__all__ = [
    "AgentEvent",
    "AgentState",
    "OrchestratorConfig",
    "SessionState",
    "Orchestrator",
]
```

---

## 8. Implementation Checklist

- [ ] Create `src/features/orchestrator/models.py` with all models
- [ ] Create `src/features/orchestrator/handlers.py` with `Orchestrator`
- [ ] Create `src/features/orchestrator/__init__.py` with exports
- [ ] Create `src/app.py` with Gradio UI
- [ ] Create `Dockerfile`
- [ ] Create/update root `README.md` for HuggingFace
- [ ] Write tests in `tests/unit/features/orchestrator/test_orchestrator.py`
- [ ] Run `uv run pytest tests/unit/features/orchestrator/ -v` — **ALL TESTS MUST PASS**
- [ ] Run `uv run python src/app.py` locally and test the UI
- [ ] Commit: `git commit -m "feat: phase 4 orchestrator and UI complete"`

---

## 9. Definition of Done

Phase 4 is **COMPLETE** when:

1. ✅ All unit tests pass
2. ✅ `uv run python src/app.py` launches Gradio UI locally
3. ✅ Can submit a question and see streaming events
4. ✅ Agent completes and generates a report
5. ✅ Dockerfile builds successfully
6. ✅ Can test full flow:

```python
import asyncio
from src.features.orchestrator.handlers import Orchestrator

async def test():
    orchestrator = Orchestrator()
    async for event in orchestrator.run("Can metformin treat Alzheimer's?"):
        print(event.to_display())

asyncio.run(test())
```

---

## 10. Deployment to HuggingFace Spaces

### Option A: Via GitHub (Recommended)

1. Push your code to GitHub
2. Create a new Space on HuggingFace
3. Connect your GitHub repo
4. Add secrets: `OPENAI_API_KEY`
5. Deploy!

### Option B: Manual Upload

1. Create a new Gradio Space on HuggingFace
2. Upload all files from `src/` and root configs
3. Add secrets in Space settings
4. Wait for build

### Verify Deployment

1. Visit your Space URL
2. Ask: "What drugs could treat long COVID?"
3. Verify streaming events appear
4. Verify final report is generated

---

**🎉 Congratulations! Phase 4 is the MVP.**

After completing Phase 4, you have a working drug repurposing research agent
that can be demonstrated at the hackathon.

**Optional Phase 5**: Improve the report synthesis with a dedicated Report agent.
