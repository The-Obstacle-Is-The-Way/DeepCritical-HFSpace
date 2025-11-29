# P3: Missing Structured Cognitive Memory (Shared Blackboard)

**Status:** OPEN
**Priority:** P3 (Architecture/Enhancement)
**Found By:** Deep Codebase Investigation
**Date:** 2025-11-29
**Spec:** [SPEC_07_LANGGRAPH_MEMORY_ARCH.md](../specs/SPEC_07_LANGGRAPH_MEMORY_ARCH.md)

## Executive Summary

DeepBoner's `AdvancedOrchestrator` has **Data Memory** (vector store for papers) but lacks **Cognitive Memory** (structured state for hypotheses, conflicts, and research plan). This causes "context drift" on long runs and prevents intelligent conflict resolution.

---

## Current Architecture (What We Have)

### 1. MagenticState (`src/agents/state.py:18-91`)
```python
class MagenticState(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    embedding_service: Any = None  # ChromaDB connection

    def add_evidence(self, new_evidence: list[Evidence]) -> int: ...
    async def search_related(self, query: str, n_results: int = 5) -> list[Evidence]: ...
```
- **What it does:** Stores Evidence objects, URL-based deduplication, semantic search via embeddings.
- **What it DOESN'T do:** Track hypotheses, conflicts, or research plan status.

### 2. EmbeddingService (`src/services/embeddings.py:29-180`)
```python
self._client = chromadb.Client()  # In-memory (Line 44)
self._collection = self._client.create_collection(
    name=f"evidence_{uuid.uuid4().hex}",  # Random name per session (Line 45-47)
    ...
)
```
- **What it does:** In-session semantic search/deduplication.
- **Limitation:** New collection per session, no persistence despite `settings.chroma_db_path` existing.

### 3. AdvancedOrchestrator (`src/orchestrators/advanced.py:51-371`)
- Uses Microsoft's `agent-framework-core` (MagenticBuilder)
- State is implicit in chat history passed between agents
- Manager decides next step by reading conversation, not structured state

---

## The Problem

| Issue | Impact | Evidence |
|-------|--------|----------|
| **No Hypothesis Tracking** | Can't update hypothesis confidence systematically | `MagenticState` has no `hypotheses` field |
| **No Conflict Detection** | Contradictory sources are ignored | No `conflicts` list to flag Source A vs Source B |
| **Context Drift** | Manager forgets original query after 50+ messages | State lives only in chat, not structured object |
| **No Plan State** | Can't pause/resume research | No `research_plan` or `next_step` tracking |

---

## The Solution: LangGraph State Graph (Nov 2025 Best Practice)

### Why LangGraph?

Based on [comprehensive analysis](https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025):

1. **Explicit State Schema:** TypedDict/Pydantic model that ALL agents read/write
2. **State Reducers:** `Annotated[List[X], operator.add]` for appending (not overwriting)
3. **HuggingFace Compatible:** Works with `langchain-huggingface` (Llama 3.1)
4. **Production-Ready:** MongoDB checkpointer for persistence, SQLite for dev

### Target Architecture

```python
# src/agents/graph/state.py (PROPOSED)
from typing import Annotated, TypedDict, Literal
import operator

class Hypothesis(TypedDict):
    id: str
    statement: str
    status: Literal["proposed", "validating", "confirmed", "refuted"]
    confidence: float
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]

class Conflict(TypedDict):
    id: str
    description: str
    source_a_id: str
    source_b_id: str
    status: Literal["open", "resolved"]
    resolution: str | None

class ResearchState(TypedDict):
    query: str  # Immutable original question
    hypotheses: Annotated[list[Hypothesis], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]
    evidence_ids: Annotated[list[str], operator.add]  # Links to ChromaDB
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: Literal["search", "judge", "resolve", "synthesize", "finish"]
    iteration_count: int
```

---

## Implementation Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `langgraph>=0.2` | State graph framework | `uv add langgraph` |
| `langchain>=0.3` | Base abstractions | `uv add langchain` |
| `langchain-huggingface` | Llama 3.1 integration | `uv add langchain-huggingface` |
| `langgraph-checkpoint-sqlite` | Dev persistence | `uv add langgraph-checkpoint-sqlite` |

**Note:** MongoDB checkpointer (`langgraph-checkpoint-mongodb`) recommended for production per [MongoDB blog](https://www.mongodb.com/company/blog/product-release-announcements/powering-long-term-memory-for-agents-langgraph).

---

## Alternative Considered: Mem0

[Mem0](https://mem0.ai/) specializes in long-term memory and [outperformed OpenAI by 26%](https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/) in benchmarks. However:

- **Mem0 excels at:** User personalization, cross-session memory
- **LangGraph excels at:** Workflow orchestration, state machines
- **Verdict:** Use LangGraph for orchestration + optionally add Mem0 for user-level memory later

---

## Quick Win (Separate from LangGraph)

Enable ChromaDB persistence in `src/services/embeddings.py:44`:
```python
# FROM:
self._client = chromadb.Client()  # In-memory

# TO:
self._client = chromadb.PersistentClient(path=settings.chroma_db_path)
```

This alone gives cross-session evidence persistence (P3_ARCHITECTURAL_GAP_EPHEMERAL_MEMORY fix).

---

## References

- [LangGraph Multi-Agent Orchestration Guide 2025](https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
- [Long-Term Agentic Memory with LangGraph](https://medium.com/@anil.jain.baba/long-term-agentic-memory-with-langgraph-824050b09852)
- [LangGraph vs LangChain 2025](https://kanerika.com/blogs/langchain-vs-langgraph/)
- [MongoDB + LangGraph Checkpointers](https://www.mongodb.com/company/blog/product-release-announcements/powering-long-term-memory-for-agents-langgraph)
- [Mem0 + LangGraph Integration](https://datacouch.io/blog/build-smarter-ai-agents-mem0-langgraph-guide/)
