# SPEC-07: Structured Cognitive Memory Architecture (LangGraph)

**Status:** APPROVED
**Priority:** HIGH (Strategic)
**Author:** DeepBoner Architecture Team
**Date:** 2025-11-29
**Last Updated:** 2025-11-29
**Related Bugs:** [P3_ARCHITECTURAL_GAP_STRUCTURED_MEMORY](../bugs/P3_ARCHITECTURAL_GAP_STRUCTURED_MEMORY.md)

---

## 1. Executive Summary

Upgrade DeepBoner's "Advanced Mode" from chat-based coordination to a **State-Driven Cognitive Architecture** using LangGraph. This enables:
- Explicit hypothesis tracking with confidence scores
- Automatic conflict detection and resolution
- Persistent research state (pause/resume)
- Context-aware decision making over long runs

---

## 2. Problem Statement

### Current Architecture Limitations

The `AdvancedOrchestrator` (`src/orchestrators/advanced.py`) uses Microsoft's `agent-framework-core` with chat-based coordination:

```python
# Current: State is IMPLICIT (chat history)
workflow = MagenticBuilder()
    .participants(searcher=..., judge=..., ...)
    .with_standard_manager(chat_client=..., max_round_count=10)
    .build()
```

| Problem | Root Cause | File Location |
|---------|------------|---------------|
| Context Drift | State lives only in chat messages | `advanced.py:126-132` |
| Conflict Blindness | No structured conflict tracking | `state.py` (no `conflicts` field) |
| No Hypothesis Management | `MagenticState` only tracks `evidence` | `state.py:21` |
| Can't Pause/Resume | No checkpointing mechanism | N/A |

### Evidence from Codebase

**MagenticState (src/agents/state.py:18-26):**
```python
class MagenticState(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    embedding_service: Any = None  # Just data, no cognitive state
```

**EmbeddingService (src/services/embeddings.py:44-47):**
```python
self._client = chromadb.Client()  # In-memory only
self._collection = self._client.create_collection(
    name=f"evidence_{uuid.uuid4().hex}",  # Random name = ephemeral
    ...
)
```

---

## 3. Solution: LangGraph State Graph

### Why LangGraph? (November 2025 Analysis)

Based on [comprehensive framework comparison](https://kanerika.com/blogs/langchain-vs-langgraph/):

| Feature | `agent-framework-core` (Current) | LangGraph (Proposed) |
|---------|----------------------------------|----------------------|
| State Management | Implicit (chat) | Explicit (TypedDict) |
| Loops/Branches | Limited | Native support |
| Checkpointing | None | SQLite/MongoDB |
| HuggingFace | Requires OpenAI format | Native `langchain-huggingface` |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ResearchState                              │
│  ┌─────────────┬──────────────┬───────────────┬──────────────┐ │
│  │   query     │  hypotheses  │   conflicts   │  next_step   │ │
│  │  (string)   │    (list)    │    (list)     │   (enum)     │ │
│  └─────────────┴──────────────┴───────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      StateGraph                                 │
│                                                                 │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│    │ SEARCH   │────▶│  JUDGE   │────▶│ RESOLVE  │              │
│    │  Node    │     │   Node   │     │   Node   │              │
│    └──────────┘     └──────────┘     └──────────┘              │
│         ▲                │                 │                    │
│         │                ▼                 │                    │
│         │          ┌──────────┐           │                    │
│         └──────────│SUPERVISOR│◀──────────┘                    │
│                    │   Node   │                                 │
│                    └──────────┘                                 │
│                          │                                      │
│                          ▼                                      │
│                    ┌──────────┐                                 │
│                    │SYNTHESIZE│                                 │
│                    │   Node   │                                 │
│                    └──────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Specification

### 4.1 State Schema

**File:** `src/agents/graph/state.py`

```python
"""Structured state for LangGraph research workflow."""
from typing import Annotated, TypedDict, Literal
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A research hypothesis with evidence tracking."""
    id: str = Field(description="Unique identifier for the hypothesis")
    statement: str = Field(description="The hypothesis statement")
    status: Literal["proposed", "validating", "confirmed", "refuted"] = Field(
        default="proposed", description="Current validation status"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str | None = Field(default=None, description="Reasoning for current status")


class Conflict(BaseModel):
    """A detected contradiction between sources."""
    id: str = Field(description="Unique identifier for the conflict")
    description: str = Field(description="Description of the contradiction")
    source_a_id: str = Field(description="ID of the first conflicting source")
    source_b_id: str = Field(description="ID of the second conflicting source")
    status: Literal["open", "resolved"] = Field(default="open")
    resolution: str | None = Field(default=None, description="Resolution explanation if resolved")


class ResearchState(TypedDict):
    """The cognitive state shared across all graph nodes.

    Uses Annotated with operator.add for list fields to enable
    additive updates (append) rather than replacement.
    """
    # Immutable context
    query: str

    # Cognitive state (The "Blackboard")
    # Note: We store these as lists of Pydantic models.
    hypotheses: Annotated[list[Hypothesis], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]

    # Evidence links (actual content stored in ChromaDB)
    evidence_ids: Annotated[list[str], operator.add]

    # Chat history (for LLM context)
    messages: Annotated[list[BaseMessage], operator.add]

    # Control flow
    next_step: Literal["search", "judge", "resolve", "synthesize", "finish"]
    iteration_count: int
    max_iterations: int
```

### 4.2 Graph Nodes

Each node is an async function that receives the state and injected dependencies.

**File:** `src/agents/graph/nodes.py`

```python
"""Graph node implementations."""
from typing import Any
from langchain_core.messages import AIMessage
from src.services.embeddings import EmbeddingService
from src.tools.search_handler import SearchHandler


async def search_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Execute search across all sources.

    Uses SearchHandler to query PubMed, ClinicalTrials, and EuropePMC.
    Deduplicates evidence using EmbeddingService.
    """
    # ... implementation ...
    return {
        "evidence_ids": new_ids,
        "messages": [AIMessage(content=message)],
    }


async def judge_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Evaluate evidence and update hypothesis confidence.

    Uses pydantic_ai Agent to generate structured HypothesisAssessment.
    """
    # ... implementation ...
    return {
        "hypotheses": new_hypotheses,
        "messages": [AIMessage(content=f"Judge: Generated {len(new_hypotheses)} hypotheses.")],
        "next_step": "resolve",
    }


async def resolve_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Handle open conflicts."""
    # ... implementation ...
    return {"messages": messages}


async def synthesize_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Generate final research report."""
    # ... implementation ...
    return {"messages": [AIMessage(content=report_markdown)], "next_step": "finish"}


async def supervisor_node(
    state: ResearchState, llm: BaseChatModel | None = None
) -> dict[str, Any]:
    """Route to next node based on state using robust Pydantic parsing.

    This is the "brain" - uses LLM to decide next action
    based on STRUCTURED STATE.
    """
    # ... implementation ...
    return {
        "next_step": decision.next_step,
        "iteration_count": state["iteration_count"] + 1,
        "messages": [AIMessage(content=f"Supervisor: {decision.reasoning}")],
    }
```

### 4.3 Graph Definition

**File:** `src/agents/graph/workflow.py`

```python
"""LangGraph workflow definition."""
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.agents.graph.state import ResearchState
from src.services.embeddings import EmbeddingService
# ... imports ...


def create_research_graph(
    llm=None,
    checkpointer=None,
    embedding_service: EmbeddingService | None = None,
) -> CompiledStateGraph:
    """Build the research state graph.

    Args:
        llm: Supervisor LLM
        checkpointer: Optional persistence layer
        embedding_service: Service for evidence storage
    """
    graph = StateGraph(ResearchState)

    # Bind dependencies using partial
    bound_supervisor = partial(supervisor_node, llm=llm) if llm else supervisor_node
    bound_search = partial(search_node, embedding_service=embedding_service)
    # ... binding other nodes ...

    # Add nodes
    graph.add_node("supervisor", bound_supervisor)
    graph.add_node("search", bound_search)
    # ...

    # ... edges ...
```

### 4.4 Orchestrator Integration

**File:** `src/orchestrators/langgraph_orchestrator.py`

```python
"""LangGraph-based orchestrator with structured state."""
from collections.abc import AsyncGenerator
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agents.graph.workflow import create_research_graph
from src.agents.graph.state import ResearchState
from src.orchestrators.base import OrchestratorProtocol
from src.utils.models import AgentEvent


class LangGraphOrchestrator(OrchestratorProtocol):
    """State-driven research orchestrator using LangGraph."""

    def __init__(
        self,
        max_iterations: int = 10,
        checkpoint_path: str | None = None,
    ):
        self._max_iterations = max_iterations
        self._checkpoint_path = checkpoint_path

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute research workflow with structured state."""
        # Setup checkpointer (SQLite for dev, MongoDB for prod)
        checkpointer = None
        if self._checkpoint_path:
            checkpointer = AsyncSqliteSaver.from_conn_string(self._checkpoint_path)

        graph = create_research_graph(checkpointer)

        # Initialize state
        initial_state: ResearchState = {
            "query": query,
            "hypotheses": [],
            "conflicts": [],
            "evidence_ids": [],
            "messages": [],
            "next_step": "search",
            "iteration_count": 0,
            "max_iterations": self._max_iterations,
        }

        yield AgentEvent(type="started", message=f"Starting research: {query}")

        # Stream through graph
        async for event in graph.astream(initial_state):
            # Convert graph events to AgentEvents
            yield self._convert_event(event)
```

---

## 5. Dependencies

### Required Packages

```toml
# pyproject.toml additions
[project.optional-dependencies]
langgraph = [
    "langgraph>=0.2.50",
    "langchain>=0.3.9",
    "langchain-core>=0.3.21",
    "langchain-huggingface>=0.1.2",
    "langgraph-checkpoint-sqlite>=2.0.0",
]
```

### Installation

```bash
# Development
uv add langgraph langchain langchain-huggingface langgraph-checkpoint-sqlite

# Production (add MongoDB checkpointer)
uv add langgraph-checkpoint-mongodb
```

### HuggingFace Model Integration

```python
# Using Llama 3.1 via HuggingFace Inference API
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    max_new_tokens=2048,
    huggingfacehub_api_token=settings.hf_token,
)
chat = ChatHuggingFace(llm=llm)
```

---

## 6. Implementation Plan (TDD)

### Phase 1: State Schema (2 hours)

1. Create `src/agents/graph/__init__.py`
2. Create `src/agents/graph/state.py` with TypedDict schemas
3. Write `tests/unit/graph/test_state.py`:
   - Test reducer behavior (operator.add)
   - Test state initialization
   - Test hypothesis/conflict type validation

### Phase 2: Graph Nodes (4 hours)

1. Create `src/agents/graph/nodes.py`
2. Adapt existing tool calls (pubmed, clinicaltrials, europepmc)
3. Write `tests/unit/graph/test_nodes.py`:
   - Test each node in isolation (mock LLM)
   - Test state update format

### Phase 3: Workflow Graph (2 hours)

1. Create `src/agents/graph/workflow.py`
2. Wire up StateGraph with conditional edges
3. Write `tests/integration/graph/test_workflow.py`:
   - Test routing logic
   - Test end-to-end with mocked nodes

### Phase 4: Orchestrator (2 hours)

1. Create `src/orchestrators/langgraph_orchestrator.py`
2. Update `src/orchestrators/factory.py` to include "langgraph" mode
3. Update `src/app.py` UI dropdown
4. Write `tests/e2e/test_langgraph_mode.py`

### Phase 5: Gradio Integration (1 hour)

1. Add "God Mode" option to Gradio dropdown
2. Test streaming events
3. Verify checkpointing (pause/resume)

---

## 7. Migration Strategy

1. **Parallel Implementation:** Build as new mode alongside existing "simple" and "magentic"
2. **UI Dropdown:** Add "God Mode (Experimental)" option
3. **Feature Flag:** Use `settings.enable_langgraph_mode` to control availability
4. **Deprecation Path:** Once stable, deprecate "magentic" mode (Q1 2026)

---

## 8. Acceptance Criteria

- [ ] `ResearchState` TypedDict defined with all fields
- [ ] All 4 nodes (search, judge, resolve, synthesize) implemented
- [ ] Supervisor routing logic works based on structured state
- [ ] Checkpointing enables pause/resume
- [ ] Works with HuggingFace Inference API (no OpenAI required)
- [ ] Integration tests pass with mocked LLM
- [ ] E2E test passes with real API call

---

## 9. References

### Primary Sources
- [LangGraph Official Docs](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph Persistence Guide](https://docs.langchain.com/oss/python/langgraph/persistence)
- [MongoDB + LangGraph Integration](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/)

### Research & Analysis
- [LangGraph Multi-Agent Orchestration 2025](https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
- [LangChain vs LangGraph Comparison](https://kanerika.com/blogs/langchain-vs-langgraph/)
- [Building Deep Research Agents](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/)
- [Mem0 + LangGraph Integration](https://blog.futuresmart.ai/ai-agents-memory-mem0-langgraph-agent-integration)
- [AI Memory Wars Benchmark](https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/)
