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


class Hypothesis(TypedDict):
    """A research hypothesis with evidence tracking."""
    id: str
    statement: str
    status: Literal["proposed", "validating", "confirmed", "refuted"]
    confidence: float  # 0.0 - 1.0
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]


class Conflict(TypedDict):
    """A detected contradiction between sources."""
    id: str
    description: str
    source_a_id: str
    source_b_id: str
    status: Literal["open", "resolved"]
    resolution: str | None


class ResearchState(TypedDict):
    """The cognitive state shared across all graph nodes.

    Uses Annotated with operator.add for list fields to enable
    additive updates (append) rather than replacement.
    """
    # Immutable context
    query: str

    # Cognitive state (the "blackboard")
    hypotheses: Annotated[list[Hypothesis], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]

    # Evidence links (actual content in ChromaDB)
    evidence_ids: Annotated[list[str], operator.add]

    # Chat history (for LLM context)
    messages: Annotated[list[BaseMessage], operator.add]

    # Control flow
    next_step: Literal["search", "judge", "resolve", "synthesize", "finish"]
    iteration_count: int
    max_iterations: int
```

### 4.2 Graph Nodes

Each node is a pure function: `(state: ResearchState) -> dict`

**File:** `src/agents/graph/nodes.py`

```python
"""Graph node implementations."""
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.pubmed import search_pubmed
from src.tools.clinicaltrials import search_clinicaltrials
from src.tools.europepmc import search_europepmc


async def search_node(state: ResearchState) -> dict:
    """Execute search across all sources.

    Returns partial state update (additive via operator.add).
    """
    query = state["query"]
    # Reuse existing tools
    results = await asyncio.gather(
        search_pubmed(query),
        search_clinicaltrials(query),
        search_europepmc(query),
    )
    new_evidence_ids = [...]  # Store in ChromaDB, return IDs
    return {
        "evidence_ids": new_evidence_ids,
        "messages": [AIMessage(content=f"Found {len(new_evidence_ids)} papers")],
    }


async def judge_node(state: ResearchState) -> dict:
    """Evaluate evidence and update hypothesis confidence.

    Key responsibility: Detect conflicts and flag them.
    """
    # LLM call to evaluate hypotheses against evidence
    # If contradiction found: add to conflicts list
    return {
        "hypotheses": updated_hypotheses,  # With new confidence scores
        "conflicts": new_conflicts,  # Any detected contradictions
        "messages": [...],
    }


async def resolve_node(state: ResearchState) -> dict:
    """Handle open conflicts via tie-breaker logic.

    Triggers targeted search or reasoning to resolve.
    """
    open_conflicts = [c for c in state["conflicts"] if c["status"] == "open"]
    # For each conflict: search for decisive evidence or make judgment call
    return {
        "conflicts": resolved_conflicts,
        "messages": [...],
    }


async def synthesize_node(state: ResearchState) -> dict:
    """Generate final research report.

    Only uses confirmed hypotheses and resolved conflicts.
    """
    confirmed = [h for h in state["hypotheses"] if h["status"] == "confirmed"]
    # Generate structured report
    return {
        "messages": [AIMessage(content=report_markdown)],
        "next_step": "finish",
    }


def supervisor_node(state: ResearchState) -> dict:
    """Route to next node based on state.

    This is the "brain" - uses LLM to decide next action
    based on STRUCTURED STATE (not just chat).
    """
    # Decision logic:
    # 1. If open conflicts exist -> "resolve"
    # 2. If hypotheses need more evidence -> "search"
    # 3. If evidence is sufficient -> "judge"
    # 4. If all hypotheses confirmed -> "synthesize"
    # 5. If max iterations -> "synthesize" (forced)
    return {"next_step": decided_step, "iteration_count": state["iteration_count"] + 1}
```

### 4.3 Graph Definition

**File:** `src/agents/graph/workflow.py`

```python
"""LangGraph workflow definition."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.graph.state import ResearchState
from src.agents.graph.nodes import (
    search_node,
    judge_node,
    resolve_node,
    synthesize_node,
    supervisor_node,
)


def create_research_graph(checkpointer=None):
    """Build the research state graph.

    Args:
        checkpointer: Optional SqliteSaver/MongoDBSaver for persistence
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("judge", judge_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("synthesize", synthesize_node)

    # Define edges (supervisor routes based on state.next_step)
    graph.add_edge("search", "supervisor")
    graph.add_edge("judge", "supervisor")
    graph.add_edge("resolve", "supervisor")
    graph.add_edge("synthesize", END)

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next_step"],
        {
            "search": "search",
            "judge": "judge",
            "resolve": "resolve",
            "synthesize": "synthesize",
            "finish": END,
        },
    )

    # Entry point
    graph.set_entry_point("supervisor")

    return graph.compile(checkpointer=checkpointer)
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
