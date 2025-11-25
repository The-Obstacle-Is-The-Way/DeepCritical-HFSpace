# Phase 5 Implementation Spec: Magentic Integration (Optional)

**Goal**: Upgrade orchestrator to use Microsoft Agent Framework's Magentic-One pattern.
**Philosophy**: "Same API, Better Engine."
**Prerequisite**: Phase 4 complete (MVP working end-to-end)

---

## 1. Why Magentic?

Magentic-One provides:
- **LLM-powered manager** that dynamically plans, selects agents, tracks progress
- **Built-in stall detection** and automatic replanning
- **Checkpointing** for pause/resume workflows
- **Event streaming** for real-time UI updates
- **Multi-agent coordination** with round limits and reset logic

This is **NOT required for MVP**. Only implement if time permits after Phase 4.

---

## 2. Architecture Alignment

### Current Phase 4 Architecture
```
User Query
    ↓
Orchestrator (while loop)
    ├── SearchHandler.execute() → Evidence
    ├── JudgeHandler.assess() → JudgeAssessment
    └── Loop/Synthesize decision
    ↓
Research Report
```

### Phase 5 Magentic Architecture
```
User Query
    ↓
MagenticBuilder
    ├── SearchAgent (wraps SearchHandler)
    ├── JudgeAgent (wraps JudgeHandler)
    └── StandardMagenticManager (LLM coordinator)
    ↓
Research Report (same output format)
```

**Key Insight**: We wrap existing handlers as `AgentProtocol` implementations. The domain logic stays the same.

---

## 3. Design for Seamless Integration

### 3.1 Protocol-Based Design (Phase 4 prep)

In Phase 4, define handlers using Protocols so they can be wrapped later:

```python
# src/orchestrator.py (Phase 4)
from typing import Protocol, List
from src.utils.models import Evidence, SearchResult, JudgeAssessment


class SearchHandlerProtocol(Protocol):
    """Protocol for search handler - can be wrapped as Agent later."""
    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        ...


class JudgeHandlerProtocol(Protocol):
    """Protocol for judge handler - can be wrapped as Agent later."""
    async def assess(self, question: str, evidence: List[Evidence]) -> JudgeAssessment:
        ...


class OrchestratorProtocol(Protocol):
    """Protocol for orchestrator - allows swapping implementations."""
    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        ...
```

### 3.2 Facade Pattern

The `Orchestrator` class is a facade. In Phase 5, we create `MagenticOrchestrator` with the same interface:

```python
# Phase 4: Simple orchestrator
orchestrator = Orchestrator(search_handler, judge_handler)

# Phase 5: Magentic orchestrator (SAME API)
orchestrator = MagenticOrchestrator(search_handler, judge_handler)

# Usage is identical
async for event in orchestrator.run("metformin alzheimer"):
    print(event.to_markdown())
```

---

## 4. Phase 5 Implementation

### 4.1 Install Agent Framework

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
magentic = [
    "agent-framework-core>=0.1.0",
]
```

### 4.2 Agent Wrappers (`src/agents/search_agent.py`)

Wrap `SearchHandler` as an `AgentProtocol`. 
**Note**: `AgentProtocol` requires `id`, `name`, `display_name`, `description`, `run`, `run_stream`, and `get_new_thread`.

```python
"""Search agent wrapper for Magentic integration."""
from typing import Any, AsyncIterable
from agent_framework import AgentProtocol, AgentRunResponse, AgentRunResponseUpdate, ChatMessage, Role, AgentThread

from src.tools.search_handler import SearchHandler
from src.utils.models import SearchResult


class SearchAgent:
    """Wraps SearchHandler as an AgentProtocol for Magentic."""

    def __init__(self, search_handler: SearchHandler):
        self._handler = search_handler
        self._id = "search-agent"
        self._name = "SearchAgent"
        self._description = "Searches PubMed and web for drug repurposing evidence"

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Execute search based on the last user message."""
        # Extract query from messages
        query = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER and msg.text:
                    query = msg.text
                    break
                elif isinstance(msg, str):
                    query = msg
                    break
        elif isinstance(messages, str):
            query = messages
        
        if not query:
            return AgentRunResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="No query provided")],
                response_id="search-no-query",
            )

        # Execute search
        result: SearchResult = await self._handler.execute(query, max_results_per_tool=10)

        # Format response
        evidence_text = "\n".join([
            f"- [{e.citation.title}]({e.citation.url}): {e.content[:200]}..."
            for e in result.evidence[:5]
        ])

        response_text = f"Found {result.total_found} sources:\n\n{evidence_text}"

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"search-{result.total_found}",
            additional_properties={"evidence": [e.model_dump() for e in result.evidence]},
        )

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper for search (search itself isn't streaming)."""
        result = await self.run(messages, thread=thread, **kwargs)
        # Yield single update with full result
        yield AgentRunResponseUpdate(
            messages=result.messages,
            response_id=result.response_id
        )

    def get_new_thread(self, **kwargs: Any) -> AgentThread:
        """Create a new thread."""
        return AgentThread(**kwargs)
```

### 4.3 Judge Agent Wrapper (`src/agents/judge_agent.py`)

```python
"""Judge agent wrapper for Magentic integration."""
from typing import Any, List, AsyncIterable
from agent_framework import AgentProtocol, AgentRunResponse, AgentRunResponseUpdate, ChatMessage, Role, AgentThread

from src.agent_factory.judges import JudgeHandler
from src.utils.models import Evidence, JudgeAssessment


class JudgeAgent:
    """Wraps JudgeHandler as an AgentProtocol for Magentic."""

    def __init__(self, judge_handler: JudgeHandler, evidence_store: dict[str, List[Evidence]]):
        self._handler = judge_handler
        self._evidence_store = evidence_store  # Shared state for evidence
        self._id = "judge-agent"
        self._name = "JudgeAgent"
        self._description = "Evaluates evidence quality and determines if sufficient for synthesis"

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Assess evidence quality."""
        # Extract original question from messages
        question = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER and msg.text:
                    question = msg.text
                    break
                elif isinstance(msg, str):
                    question = msg
                    break
        elif isinstance(messages, str):
            question = messages

        # Get evidence from shared store
        evidence = self._evidence_store.get("current", [])

        # Assess
        assessment: JudgeAssessment = await self._handler.assess(question, evidence)

        # Format response
        response_text = f"""## Assessment

**Sufficient**: {assessment.sufficient}
**Confidence**: {assessment.confidence:.0%}
**Recommendation**: {assessment.recommendation}

### Scores
- Mechanism: {assessment.details.mechanism_score}/10
- Clinical: {assessment.details.clinical_evidence_score}/10

### Reasoning
{assessment.reasoning}
"""

        if assessment.next_search_queries:
            response_text += f"\n### Next Queries\n" + "\n".join(
                f"- {q}" for q in assessment.next_search_queries
            )

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"judge-{assessment.recommendation}",
            additional_properties={"assessment": assessment.model_dump()},
        )

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper for judge."""
        result = await self.run(messages, thread=thread, **kwargs)
        yield AgentRunResponseUpdate(
            messages=result.messages,
            response_id=result.response_id
        )
        
    def get_new_thread(self, **kwargs: Any) -> AgentThread:
        """Create a new thread."""
        return AgentThread(**kwargs)
```

### 4.4 Magentic Orchestrator (`src/orchestrator_magentic.py`)

```python
"""Magentic-based orchestrator for DeepCritical."""
from typing import AsyncGenerator, List
import structlog

from agent_framework import (
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticAgentMessageEvent,
    MagenticOrchestratorMessageEvent,
    MagenticAgentDeltaEvent,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

from src.agents.search_agent import SearchAgent
from src.agents.judge_agent import JudgeAgent
from src.tools.search_handler import SearchHandler
from src.agent_factory.judges import JudgeHandler
from src.utils.models import AgentEvent, Evidence

logger = structlog.get_logger()


class MagenticOrchestrator:
    """
    Magentic-based orchestrator - same API as Orchestrator.

    Uses Microsoft Agent Framework's MagenticBuilder for multi-agent coordination.
    """

    def __init__(
        self,
        search_handler: SearchHandler,
        judge_handler: JudgeHandler,
        max_rounds: int = 10,
    ):
        self._search_handler = search_handler
        self._judge_handler = judge_handler
        self._max_rounds = max_rounds
        self._evidence_store: dict[str, List[Evidence]] = {"current": []}

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the Magentic workflow - same API as simple Orchestrator.

        Yields AgentEvent objects for real-time UI updates.
        """
        logger.info("Starting Magentic orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research (Magentic mode): {query}",
            iteration=0,
        )

        # Create agent wrappers
        search_agent = SearchAgent(self._search_handler)
        judge_agent = JudgeAgent(self._judge_handler, self._evidence_store)

        # Build Magentic workflow
        # Note: MagenticBuilder.participants takes named arguments for agent instances
        workflow = (
            MagenticBuilder()
            .participants(
                searcher=search_agent,
                judge=judge_agent,
            )
            .with_standard_manager(
                chat_client=OpenAIChatClient(),
                max_round_count=self._max_rounds,
                max_stall_count=3,
                max_reset_count=2,
            )
            .build()
        )

        # Task instruction for the manager
        task = f"""Research drug repurposing opportunities for: {query}

Instructions:
1. Use SearcherAgent to find evidence from PubMed and web
2. Use JudgeAgent to evaluate if evidence is sufficient
3. If JudgeAgent says "continue", search with refined queries
4. If JudgeAgent says "synthesize", provide final synthesis
5. Stop when synthesis is ready or max rounds reached

Focus on finding:
- Mechanism of action evidence
- Clinical/preclinical studies
- Specific drug candidates
"""

        iteration = 0
        try:
            # workflow.run_stream returns an async generator of workflow events
            async for event in workflow.run_stream(task):
                if isinstance(event, MagenticOrchestratorMessageEvent):
                    # Manager events (planning, instruction, ledger)
                    message_text = event.message.text if event.message else ""
                    yield AgentEvent(
                        type="judging",
                        message=f"Manager ({event.kind}): {message_text[:100]}...",
                        iteration=iteration,
                    )

                elif isinstance(event, MagenticAgentMessageEvent):
                    # Complete agent response
                    iteration += 1
                    agent_name = event.agent_id or "unknown"
                    msg_text = event.message.text if event.message else ""

                    if "search" in agent_name.lower():
                        # Check if we found evidence (based on SearchAgent logic)
                        # In a real implementation we might extract metadata
                        yield AgentEvent(
                            type="search_complete",
                            message=f"Search agent: {msg_text[:100]}...",
                            iteration=iteration,
                        )
                    elif "judge" in agent_name.lower():
                        yield AgentEvent(
                            type="judge_complete",
                            message=f"Judge agent: {msg_text[:100]}...",
                            iteration=iteration,
                        )

                elif isinstance(event, MagenticFinalResultEvent):
                    # Final workflow result
                    final_text = event.message.text if event.message else "No result"
                    yield AgentEvent(
                        type="complete",
                        message=final_text,
                        data={"iterations": iteration},
                        iteration=iteration,
                    )

                elif isinstance(event, MagenticAgentDeltaEvent):
                    # Streaming token chunks from agents (optional "typing" effect)
                    # Only emit if we have actual text content
                    if event.text:
                        yield AgentEvent(
                            type="streaming",
                            message=event.text,
                            data={"agent_id": event.agent_id},
                            iteration=iteration,
                        )

                elif isinstance(event, WorkflowOutputEvent):
                    # Alternative final output event
                    if event.data:
                        yield AgentEvent(
                            type="complete",
                            message=str(event.data),
                            iteration=iteration,
                        )

        except Exception as e:
            logger.error("Magentic workflow failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Workflow error: {str(e)}",
                iteration=iteration,
            )
```

### 4.5 Factory Pattern (`src/orchestrator_factory.py`)

Allow switching between implementations:

```python
"""Factory for creating orchestrators."""
from typing import Literal

from src.orchestrator import Orchestrator
from src.tools.search_handler import SearchHandler
from src.agent_factory.judges import JudgeHandler
from src.utils.models import OrchestratorConfig


def create_orchestrator(
    search_handler: SearchHandler,
    judge_handler: JudgeHandler,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic"] = "simple",
):
    """
    Create an orchestrator instance.

    Args:
        search_handler: The search handler
        judge_handler: The judge handler
        config: Optional configuration
        mode: "simple" for Phase 4 loop, "magentic" for Phase 5 multi-agent

    Returns:
        Orchestrator instance (same interface regardless of mode)
    """
    if mode == "magentic":
        try:
            from src.orchestrator_magentic import MagenticOrchestrator
            return MagenticOrchestrator(
                search_handler=search_handler,
                judge_handler=judge_handler,
                max_rounds=config.max_iterations if config else 10,
            )
        except ImportError:
            # Fallback to simple if agent-framework not installed
            pass

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )
```

---

## 5. Directory Structure After Phase 5

```
src/
├── app.py                      # Gradio UI (unchanged)
├── orchestrator.py             # Phase 4 simple orchestrator
├── orchestrator_magentic.py    # Phase 5 Magentic orchestrator
├── orchestrator_factory.py     # Factory to switch implementations
├── agents/                     # NEW: Agent wrappers
│   ├── __init__.py
│   ├── search_agent.py         # SearchHandler as AgentProtocol
│   └── judge_agent.py          # JudgeHandler as AgentProtocol
├── agent_factory/
│   └── judges.py               # JudgeHandler (unchanged)
├── tools/
│   ├── pubmed.py               # PubMed tool (unchanged)
│   ├── websearch.py            # Web tool (unchanged)
│   └── search_handler.py       # SearchHandler (unchanged)
└── utils/
    └── models.py               # Models (unchanged)
```

---

## 6. Implementation Checklist

- [ ] Ensure Phase 4 uses Protocol-based handler interfaces
- [ ] Add `agent-framework-core` to optional dependencies
- [ ] Create `src/agents/` directory
- [ ] Implement `SearchAgent` wrapper
- [ ] Implement `JudgeAgent` wrapper
- [ ] Implement `MagenticOrchestrator`
- [ ] Implement `orchestrator_factory.py`
- [ ] Add tests for agent wrappers
- [ ] Test Magentic flow end-to-end
- [ ] Update `src/app.py` to use factory with mode toggle

---

## 7. Definition of Done

Phase 5 is **COMPLETE** when:

1. All Phase 4 tests still pass (no regression)
2. `MagenticOrchestrator` has same API as `Orchestrator`
3. Can switch between modes via factory:

```python
# Simple mode (Phase 4)
orchestrator = create_orchestrator(search, judge, mode="simple")

# Magentic mode (Phase 5)
orchestrator = create_orchestrator(search, judge, mode="magentic")

# Same usage!
async for event in orchestrator.run("metformin alzheimer"):
    print(event.to_markdown())
```

4. UI works with both modes
5. Graceful fallback if agent-framework not installed

---

## 8. When to Implement

**Priority**: LOW (optional enhancement)

Implement ONLY after:
1. ✅ Phase 1: Foundation
2. ✅ Phase 2: Search
3. ✅ Phase 3: Judge
4. ✅ Phase 4: Orchestrator + UI (MVP SHIPPED)

If hackathon deadline is approaching, **SKIP Phase 5**. Ship the MVP.

---

## 9. Benefits of This Design

1. **No breaking changes** - Phase 4 code works unchanged
2. **Same API** - `run()` returns `AsyncGenerator[AgentEvent, None]`
3. **Gradual adoption** - Optional dependency, factory fallback
4. **Testable** - Each component can be tested independently
5. **Aligns with Tonic's vision** - Uses Microsoft Agent Framework patterns

---

## 10. Reference

- Microsoft Agent Framework: `reference_repos/agent-framework/`
- Magentic samples: `reference_repos/agent-framework/python/samples/getting_started/workflows/orchestration/magentic.py`
- AgentProtocol: `reference_repos/agent-framework/python/packages/core/agent_framework/_agents.py`
