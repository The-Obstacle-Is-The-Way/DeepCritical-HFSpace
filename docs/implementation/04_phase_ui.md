# Phase 4 Implementation Spec: Orchestrator & UI

**Goal**: Connect the Brain and the Body, then give it a Face.
**Philosophy**: "Streaming is Trust."
**Estimated Effort**: 4-5 hours
**Prerequisite**: Phases 1-3 complete

---

## 1. The Slice Definition

This slice connects:
1. **Orchestrator**: The main loop calling `SearchHandler` → `JudgeHandler`.
2. **Synthesis**: Generate a final markdown report.
3. **UI**: Gradio streaming chat interface.
4. **Deployment**: Dockerfile + HuggingFace Spaces config.

**Files**:
- `src/utils/models.py`: Add AgentState, AgentEvent
- `src/orchestrator.py`: Main agent loop
- `src/app.py`: Gradio UI
- `Dockerfile`: Container build
- `README.md`: HuggingFace Space config (at root)

---

## 2. Models (`src/utils/models.py`)

Add these to the existing models file (after JudgeAssessment):

```python
# Add to src/utils/models.py (after JudgeAssessment class)

from enum import Enum
from typing import Any


class AgentState(str, Enum):
    """States of the agent during execution."""

    INITIALIZING = "initializing"
    SEARCHING = "searching"
    JUDGING = "judging"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


class AgentEvent(BaseModel):
    """An event emitted during agent execution (for streaming UI)."""

    state: AgentState = Field(description="Current agent state")
    message: str = Field(description="Human-readable status message")
    iteration: int = Field(default=0, ge=0, description="Current iteration number")
    data: dict[str, Any] | None = Field(
        default=None,
        description="Optional payload (e.g., evidence count, assessment scores)"
    )

    def to_display(self) -> str:
        """Format for UI display."""
        icon = {
            AgentState.INITIALIZING: "🔄",
            AgentState.SEARCHING: "🔍",
            AgentState.JUDGING: "⚖️",
            AgentState.SYNTHESIZING: "📝",
            AgentState.COMPLETE: "✅",
            AgentState.ERROR: "❌",
        }.get(self.state, "▶️")
        return f"{icon} **[{self.state.value.upper()}]** {self.message}"


class AgentResult(BaseModel):
    """Final result from the agent."""

    question: str = Field(description="The original research question")
    report: str = Field(description="The synthesized markdown report")
    evidence_count: int = Field(description="Total evidence items collected")
    iterations: int = Field(description="Number of search iterations")
    candidates: list["DrugCandidate"] = Field(
        default_factory=list,
        description="Drug candidates identified"
    )
    quality_score: int = Field(default=0, description="Final quality score")
```

---

## 3. Orchestrator (`src/orchestrator.py`)

```python
"""Main agent orchestrator - coordinates Search → Judge → Synthesize loop."""
import structlog
from typing import AsyncGenerator
from pydantic_ai import Agent

from src.utils.config import settings
from src.utils.exceptions import DeepCriticalError
from src.utils.models import (
    AgentEvent,
    AgentState,
    AgentResult,
    Evidence,
    JudgeAssessment,
)
from src.tools.pubmed import PubMedTool
from src.tools.websearch import WebTool
from src.tools.search_handler import SearchHandler
from src.agent_factory.judges import JudgeHandler
from src.prompts.judge import build_synthesis_prompt

logger = structlog.get_logger()


def _get_model_string() -> str:
    """Get the PydanticAI model string from settings."""
    provider = settings.llm_provider
    model = settings.llm_model
    if ":" in model:
        return model
    return f"{provider}:{model}"


# Synthesis agent for generating the final report
synthesis_agent = Agent(
    model=_get_model_string(),
    result_type=str,
    system_prompt="""You are a biomedical research report writer.
Generate comprehensive, well-structured markdown reports on drug repurposing research.
Include citations, mechanisms of action, and recommendations.
Be objective and scientific.""",
)


class Orchestrator:
    """Main orchestrator for the DeepCritical agent."""

    def __init__(
        self,
        search_handler: SearchHandler | None = None,
        judge_handler: JudgeHandler | None = None,
        max_iterations: int | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            search_handler: Optional SearchHandler (for testing).
            judge_handler: Optional JudgeHandler (for testing).
            max_iterations: Max search iterations (default from settings).
        """
        self.search_handler = search_handler or SearchHandler([
            PubMedTool(),
            WebTool(),
        ])
        self.judge_handler = judge_handler or JudgeHandler()
        self.max_iterations = max_iterations or settings.max_iterations

    async def run(self, question: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop, yielding events for streaming UI.

        Args:
            question: The research question to investigate.

        Yields:
            AgentEvent objects for each state change.
        """
        logger.info("orchestrator_starting", question=question[:100])

        # Track state
        all_evidence: list[Evidence] = []
        iteration = 0
        last_assessment: JudgeAssessment | None = None

        try:
            # Initial event
            yield AgentEvent(
                state=AgentState.INITIALIZING,
                message=f"Starting research on: {question[:100]}...",
                iteration=0,
            )

            # Main search → judge loop
            while iteration < self.max_iterations:
                iteration += 1

                # === SEARCH PHASE ===
                yield AgentEvent(
                    state=AgentState.SEARCHING,
                    message=f"Searching (iteration {iteration}/{self.max_iterations})...",
                    iteration=iteration,
                )

                # Determine search query
                if last_assessment and last_assessment.next_search_queries:
                    # Use judge's suggested queries
                    search_query = last_assessment.next_search_queries[0]
                else:
                    # Use original question
                    search_query = question

                # Execute search
                search_result = await self.search_handler.execute(
                    search_query,
                    max_results_per_tool=10,
                )

                # Accumulate evidence (deduplicate by URL)
                existing_urls = {e.citation.url for e in all_evidence}
                new_evidence = [
                    e for e in search_result.evidence
                    if e.citation.url not in existing_urls
                ]
                all_evidence.extend(new_evidence)

                yield AgentEvent(
                    state=AgentState.SEARCHING,
                    message=f"Found {len(new_evidence)} new items ({len(all_evidence)} total)",
                    iteration=iteration,
                    data={
                        "new_count": len(new_evidence),
                        "total_count": len(all_evidence),
                        "sources": search_result.sources_searched,
                    },
                )

                # === JUDGE PHASE ===
                yield AgentEvent(
                    state=AgentState.JUDGING,
                    message="Evaluating evidence quality...",
                    iteration=iteration,
                )

                last_assessment = await self.judge_handler.assess(
                    question,
                    all_evidence[-20:],  # Evaluate most recent 20 items
                )

                yield AgentEvent(
                    state=AgentState.JUDGING,
                    message=(
                        f"Quality: {last_assessment.overall_quality_score}/10, "
                        f"Coverage: {last_assessment.coverage_score}/10"
                    ),
                    iteration=iteration,
                    data={
                        "quality_score": last_assessment.overall_quality_score,
                        "coverage_score": last_assessment.coverage_score,
                        "sufficient": last_assessment.sufficient,
                        "candidates": len(last_assessment.candidates),
                    },
                )

                # Check if we should stop
                if not await self.judge_handler.should_continue(last_assessment):
                    logger.info(
                        "orchestrator_sufficient_evidence",
                        iteration=iteration,
                        evidence_count=len(all_evidence),
                    )
                    break

                # Log why we're continuing
                if last_assessment.gaps:
                    logger.info(
                        "orchestrator_continuing",
                        gaps=last_assessment.gaps[:3],
                        next_query=last_assessment.next_search_queries[:1],
                    )

            # === SYNTHESIS PHASE ===
            yield AgentEvent(
                state=AgentState.SYNTHESIZING,
                message="Generating research report...",
                iteration=iteration,
            )

            report = await self._synthesize_report(
                question,
                all_evidence,
                last_assessment,
            )

            # === COMPLETE ===
            yield AgentEvent(
                state=AgentState.COMPLETE,
                message="Research complete!",
                iteration=iteration,
                data={
                    "evidence_count": len(all_evidence),
                    "candidates": (
                        len(last_assessment.candidates) if last_assessment else 0
                    ),
                    "report_length": len(report),
                },
            )

            # Yield final report as special event
            yield AgentEvent(
                state=AgentState.COMPLETE,
                message=report,  # The report itself
                iteration=iteration,
                data={"is_report": True},
            )

        except Exception as e:
            logger.error("orchestrator_error", error=str(e))
            yield AgentEvent(
                state=AgentState.ERROR,
                message=f"Error: {str(e)}",
                iteration=iteration,
            )
            raise DeepCriticalError(f"Orchestrator failed: {e}") from e

    async def _synthesize_report(
        self,
        question: str,
        evidence: list[Evidence],
        assessment: JudgeAssessment | None,
    ) -> str:
        """Generate the final research report.

        Args:
            question: The research question.
            evidence: All collected evidence.
            assessment: The final judge assessment.

        Returns:
            Markdown formatted report.
        """
        if not assessment:
            # Fallback assessment
            assessment = JudgeAssessment(
                sufficient=True,
                recommendation="synthesize",
                reasoning="Manual synthesis requested.",
                overall_quality_score=5,
                coverage_score=5,
            )

        # Build synthesis prompt
        prompt = build_synthesis_prompt(question, assessment, evidence)

        # Generate report
        result = await synthesis_agent.run(prompt)

        return result.data

    async def run_to_completion(self, question: str) -> AgentResult:
        """Run the agent and return final result (non-streaming).

        Args:
            question: The research question.

        Returns:
            AgentResult with report and metadata.
        """
        report = ""
        evidence_count = 0
        iterations = 0
        candidates = []
        quality_score = 0

        async for event in self.run(question):
            iterations = event.iteration
            if event.data:
                if event.data.get("is_report"):
                    report = event.message
                if "evidence_count" in event.data:
                    evidence_count = event.data["evidence_count"]
                if "candidates" in event.data:
                    candidates = event.data.get("candidates", [])
                if "quality_score" in event.data:
                    quality_score = event.data["quality_score"]

        return AgentResult(
            question=question,
            report=report,
            evidence_count=evidence_count,
            iterations=iterations,
            candidates=candidates,
            quality_score=quality_score,
        )
```

---

## 4. UI (`src/app.py`)

```python
"""Gradio UI for DeepCritical agent."""
import gradio as gr
from typing import AsyncGenerator

from src.orchestrator import Orchestrator
from src.utils.models import AgentEvent, AgentState


async def chat(
    message: str,
    history: list[list[str]],
) -> AsyncGenerator[str, None]:
    """Process a chat message and stream responses.

    Args:
        message: User's research question.
        history: Chat history (not used, fresh agent each time).

    Yields:
        Streaming response text.
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    orchestrator = Orchestrator()
    full_response = ""

    try:
        async for event in orchestrator.run(message):
            # Format event for display
            if event.data and event.data.get("is_report"):
                # Final report - yield as-is
                full_response = event.message
                yield full_response
            else:
                # Status update
                status = event.to_display()
                full_response += f"\n{status}"
                yield full_response

    except Exception as e:
        yield f"\n❌ **Error**: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(
        title="DeepCritical - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🧬 DeepCritical
            ## AI-Powered Drug Repurposing Research Agent

            Enter a research question about drug repurposing to get started.
            The agent will search PubMed and the web, evaluate evidence quality,
            and generate a comprehensive research report.

            **Example questions:**
            - "Can metformin be repurposed to treat Alzheimer's disease?"
            - "What existing drugs might help treat long COVID fatigue?"
            - "Are there diabetes drugs that could treat Parkinson's?"
            """
        )

        chatbot = gr.Chatbot(
            label="Research Assistant",
            height=600,
            show_copy_button=True,
            render_markdown=True,
        )

        msg = gr.Textbox(
            label="Research Question",
            placeholder="e.g., Can metformin be repurposed to treat Alzheimer's disease?",
            lines=2,
            max_lines=5,
        )

        with gr.Row():
            submit_btn = gr.Button("🔬 Research", variant="primary")
            clear_btn = gr.Button("🗑️ Clear")

        # Examples
        gr.Examples(
            examples=[
                "Can metformin be repurposed to treat Alzheimer's disease?",
                "What existing drugs might help treat long COVID fatigue?",
                "Are there cancer drugs that could treat autoimmune diseases?",
                "Can diabetes medications help with heart failure?",
            ],
            inputs=msg,
        )

        # Event handlers
        async def respond(message: str, chat_history: list):
            """Handle user message and stream response."""
            chat_history = chat_history or []
            chat_history.append([message, ""])

            async for response in chat(message, chat_history):
                chat_history[-1][1] = response
                yield "", chat_history

        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        clear_btn.click(lambda: (None, []), outputs=[msg, chatbot])

        gr.Markdown(
            """
            ---
            **Disclaimer**: This tool is for research purposes only.
            Always consult healthcare professionals for medical decisions.

            Built with ❤️ using PydanticAI, Gradio, and Claude.
            """
        )

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
```

---

## 5. Deployment Files

### `Dockerfile`

```dockerfile
# DeepCritical Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync --no-dev

# Copy source code
COPY src/ src/

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the app
CMD ["uv", "run", "python", "src/app.py"]
```

### `README.md` (HuggingFace Space Config)

> Note: This is for the HuggingFace Space, placed at project root.

```markdown
---
title: DeepCritical
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
python_version: 3.11
app_file: src/app.py
pinned: false
license: mit
---

# DeepCritical - Drug Repurposing Research Agent

An AI-powered research assistant that searches biomedical literature to identify
drug repurposing opportunities.

## Features

- 🔍 Searches PubMed and web sources
- ⚖️ Evaluates evidence quality using AI
- 📝 Generates comprehensive research reports
- 💊 Identifies drug repurposing candidates

## How to Use

1. Enter a research question about drug repurposing
2. Wait for the agent to search and analyze literature
3. Review the generated research report

## Example Questions

- "Can metformin be repurposed to treat Alzheimer's disease?"
- "What existing drugs might help treat long COVID?"
- "Are there diabetes drugs that could treat Parkinson's?"

## Technical Details

Built with:
- PydanticAI for structured LLM outputs
- PubMed E-utilities for biomedical search
- DuckDuckGo for web search
- Gradio for the interface

## Disclaimer

This tool is for research purposes only. Always consult healthcare professionals.
```

---

## 6. TDD Workflow

### Test File: `tests/unit/test_orchestrator.py`

```python
"""Unit tests for Orchestrator."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestOrchestrator:
    """Tests for Orchestrator."""

    @pytest.mark.asyncio
    async def test_run_yields_events(self, mocker):
        """Orchestrator.run should yield AgentEvents."""
        from src.orchestrator import Orchestrator
        from src.utils.models import (
            AgentEvent,
            AgentState,
            SearchResult,
            JudgeAssessment,
            Evidence,
            Citation,
        )

        # Mock search handler
        mock_search = MagicMock()
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
            sources_searched=["pubmed", "web"],
            total_found=1,
        ))

        # Mock judge handler - return "synthesize" immediately
        mock_judge = MagicMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Good evidence.",
            overall_quality_score=8,
            coverage_score=8,
            candidates=[],
        ))
        mock_judge.should_continue = AsyncMock(return_value=False)

        # Mock synthesis
        mocker.patch(
            "src.orchestrator.synthesis_agent.run",
            new=AsyncMock(return_value=MagicMock(data="# Test Report"))
        )

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            max_iterations=3,
        )

        events = []
        async for event in orchestrator.run("test question"):
            events.append(event)

        # Should have multiple events
        assert len(events) >= 4  # init, search, judge, complete

        # Check state progression
        states = [e.state for e in events]
        assert AgentState.INITIALIZING in states
        assert AgentState.SEARCHING in states
        assert AgentState.JUDGING in states
        assert AgentState.COMPLETE in states

    @pytest.mark.asyncio
    async def test_run_respects_max_iterations(self, mocker):
        """Orchestrator should stop at max_iterations."""
        from src.orchestrator import Orchestrator
        from src.utils.models import SearchResult, JudgeAssessment, Evidence, Citation

        # Mock search
        mock_search = MagicMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[
                Evidence(
                    content="Test",
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

        # Mock judge - always say "continue"
        mock_judge = MagicMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=False,
            recommendation="continue",
            reasoning="Need more evidence.",
            overall_quality_score=4,
            coverage_score=4,
            next_search_queries=["more research"],
        ))
        mock_judge.should_continue = AsyncMock(return_value=True)

        # Mock synthesis
        mocker.patch(
            "src.orchestrator.synthesis_agent.run",
            new=AsyncMock(return_value=MagicMock(data="# Report"))
        )

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            max_iterations=2,  # Low limit
        )

        iterations_seen = set()
        async for event in orchestrator.run("test"):
            iterations_seen.add(event.iteration)

        # Should not exceed max_iterations
        assert max(iterations_seen) <= 2

    @pytest.mark.asyncio
    async def test_run_handles_errors(self, mocker):
        """Orchestrator should yield error event on failure."""
        from src.orchestrator import Orchestrator
        from src.utils.models import AgentState
        from src.utils.exceptions import DeepCriticalError

        # Mock search to raise error
        mock_search = MagicMock()
        mock_search.execute = AsyncMock(side_effect=Exception("Search failed"))

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=MagicMock(),
            max_iterations=3,
        )

        events = []
        with pytest.raises(DeepCriticalError):
            async for event in orchestrator.run("test"):
                events.append(event)

        # Should have error event
        error_events = [e for e in events if e.state == AgentState.ERROR]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_run_to_completion_returns_result(self, mocker):
        """run_to_completion should return AgentResult."""
        from src.orchestrator import Orchestrator
        from src.utils.models import SearchResult, JudgeAssessment, AgentResult, Evidence, Citation

        # Mock search
        mock_search = MagicMock()
        mock_search.execute = AsyncMock(return_value=SearchResult(
            query="test",
            evidence=[
                Evidence(
                    content="Test",
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

        # Mock judge
        mock_judge = MagicMock()
        mock_judge.assess = AsyncMock(return_value=JudgeAssessment(
            sufficient=True,
            recommendation="synthesize",
            reasoning="Good.",
            overall_quality_score=8,
            coverage_score=8,
        ))
        mock_judge.should_continue = AsyncMock(return_value=False)

        # Mock synthesis
        mocker.patch(
            "src.orchestrator.synthesis_agent.run",
            new=AsyncMock(return_value=MagicMock(data="# Test Report\n\nContent here."))
        )

        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
        )

        result = await orchestrator.run_to_completion("test question")

        assert isinstance(result, AgentResult)
        assert result.question == "test question"
        assert "Test Report" in result.report


class TestAgentEvent:
    """Tests for AgentEvent model."""

    def test_to_display_formats_correctly(self):
        """to_display should format event with icon."""
        from src.utils.models import AgentEvent, AgentState

        event = AgentEvent(
            state=AgentState.SEARCHING,
            message="Searching PubMed...",
            iteration=1,
        )

        display = event.to_display()

        assert "🔍" in display
        assert "SEARCHING" in display
        assert "Searching PubMed" in display

    def test_to_display_handles_all_states(self):
        """to_display should handle all AgentState values."""
        from src.utils.models import AgentEvent, AgentState

        for state in AgentState:
            event = AgentEvent(state=state, message="Test")
            display = event.to_display()
            assert state.value.upper() in display
```

---

## 7. Implementation Checklist

- [ ] Add `AgentState`, `AgentEvent`, `AgentResult` models to `src/utils/models.py`
- [ ] Implement `src/orchestrator.py` (complete Orchestrator class)
- [ ] Implement `src/app.py` (complete Gradio UI)
- [ ] Create `Dockerfile`
- [ ] Update root `README.md` for HuggingFace Spaces
- [ ] Write tests in `tests/unit/test_orchestrator.py`
- [ ] Run `uv run pytest tests/unit/test_orchestrator.py -v` — **ALL TESTS MUST PASS**
- [ ] Run `uv run ruff check src` — **NO ERRORS**
- [ ] Run `uv run mypy src` — **NO ERRORS**
- [ ] Run `uv run python src/app.py` — **VERIFY UI LOADS**
- [ ] Test with real query locally
- [ ] Build Docker image: `docker build -t deepcritical .`
- [ ] Commit: `git commit -m "feat: phase 4 orchestrator and UI complete"`

---

## 8. Definition of Done

Phase 4 is **COMPLETE** when:

1. ✅ All unit tests pass
2. ✅ Orchestrator yields streaming AgentEvents
3. ✅ Orchestrator respects max_iterations
4. ✅ Graceful error handling with error events
5. ✅ Gradio UI renders streaming updates
6. ✅ Ruff and mypy pass with no errors
7. ✅ Docker builds successfully
8. ✅ Manual smoke test works:

```bash
# Run locally
uv run python src/app.py

# Open http://localhost:7860 and test:
# "What existing drugs might help treat long COVID fatigue?"

# Verify:
# - Status updates stream in real-time
# - Final report is formatted as markdown
# - No errors in console
```

---

## 9. Deployment to HuggingFace Spaces

### Option A: Via GitHub (Recommended)

1. Push your code to GitHub
2. Create a new Space on HuggingFace (Gradio SDK)
3. Connect your GitHub repo
4. Add secrets in Space settings:
   - `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)
5. Deploy automatically on push

### Option B: Manual Upload

1. Create new Gradio Space on HuggingFace
2. Upload all files:
   - `src/` directory
   - `pyproject.toml`
   - `README.md`
3. Add secrets in Space settings
4. Wait for build

### Verify Deployment

1. Visit your Space URL
2. Ask: "What drugs could treat long COVID?"
3. Verify:
   - Streaming events appear
   - Final report is generated
   - No timeout errors

---

## 10. Post-MVP Enhancements (Optional)

After completing the MVP, consider:

1. **RAG Enhancement**: Add vector storage for evidence retrieval
2. **Clinical Trials**: Integrate ClinicalTrials.gov API
3. **Drug Database**: Add DrugBank or ChEMBL integration
4. **Report Export**: Add PDF/DOCX export
5. **History**: Save research sessions
6. **Multi-turn**: Allow follow-up questions

---

**🎉 Congratulations! Phase 4 is the MVP.**

After completing Phase 4, you have a working drug repurposing research agent
that can be demonstrated at the hackathon!
