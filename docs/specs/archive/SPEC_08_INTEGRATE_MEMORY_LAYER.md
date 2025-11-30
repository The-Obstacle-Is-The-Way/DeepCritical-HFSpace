# SPEC 08: Integrate Memory Layer into All Modes

**Status:** APPROVED
**Priority:** P1 (Post-Hackathon)
**Author:** Architecture Team
**Date:** 2025-11-29
**Depends On:** SPEC_07 (LangGraph Memory - IMPLEMENTED)
**Related Issue:** #73

---

## 1. Executive Summary

Integrate the structured memory layer (built in SPEC_07 as "God Mode") into Simple and Advanced modes. Remove the separate "God Mode" - memory becomes a shared capability, not a separate mode.

**Before (current - accidental):**
```
Simple Mode     → No memory
Advanced Mode   → Chat-based memory
God Mode        → Structured memory  ← ISOLATED
```

**After (target):**
```
Simple Mode     → Structured memory ✓
Advanced Mode   → Structured memory ✓
(God Mode removed from UI)
```

---

## 2. What SPEC_07 Built (Already Done)

| Component | File | Status |
|-----------|------|--------|
| `ResearchState` TypedDict | `src/agents/graph/state.py` | ✅ Done |
| `Hypothesis` model | `src/agents/graph/state.py` | ✅ Done |
| `Conflict` model | `src/agents/graph/state.py` | ✅ Done |
| `EmbeddingService` | `src/services/embeddings.py` | ✅ Done |
| Hypothesis conversion | `src/agents/graph/nodes.py` | ✅ Done |

**This is the memory layer. It works. We just need to wire it into Simple and Advanced modes.**

---

## 3. Integration Plan

### Phase 1: Create Shared Memory Service

Extract the memory logic from LangGraph nodes into a standalone service.

**New File:** `src/services/research_memory.py`

```python
"""Shared research memory layer for all orchestration modes."""

from typing import Literal

from src.agents.graph.state import Conflict, Hypothesis
from src.services.embeddings import EmbeddingService
from src.utils.models import Citation, Evidence


class ResearchMemory:
    """Shared cognitive state for research workflows.

    This is the memory layer that ALL modes use.
    It mimics the LangGraph state management but for manual orchestration.
    """

    def __init__(self, query: str, embedding_service: EmbeddingService | None = None):
        self.query = query
        self.hypotheses: list[Hypothesis] = []
        self.conflicts: list[Conflict] = []
        self.evidence_ids: list[str] = []
        self.iteration_count: int = 0
        
        # Injected service
        self._embedding_service = embedding_service or EmbeddingService()

    async def store_evidence(self, evidence: list[Evidence]) -> list[str]:
        """Store evidence and return new IDs (deduped)."""
        if not self._embedding_service:
            return []

        unique = await self._embedding_service.deduplicate(evidence)
        new_ids = []

        for ev in unique:
            ev_id = ev.citation.url
            await self._embedding_service.add_evidence(
                evidence_id=ev_id,
                content=ev.content,
                metadata={
                    "source": ev.citation.source,
                    "title": ev.citation.title,
                    "date": ev.citation.date,
                    "authors": ",".join(ev.citation.authors or []),
                    "url": ev.citation.url,
                },
            )
            new_ids.append(ev_id)

        self.evidence_ids.extend(new_ids)
        return new_ids

    async def get_relevant_evidence(self, n: int = 20) -> list[Evidence]:
        """Retrieve relevant evidence for current query."""
        if not self._embedding_service:
            return []
            
        results = await self._embedding_service.search_similar(self.query, n_results=n)
        evidence_list = []
        
        for r in results:
            meta = r.get("metadata", {})
            authors_str = meta.get("authors", "")
            authors = authors_str.split(",") if authors_str else []
            
            # Reconstruct Evidence object
            # Note: SourceName validation might be needed, defaulting to 'web' or similar if unknown
            source_raw = meta.get("source", "web")
            
            citation = Citation(
                source=source_raw, # type: ignore
                title=meta.get("title", "Unknown"),
                url=meta.get("url", r["id"]),
                date=meta.get("date", "Unknown"),
                authors=authors
            )
            
            evidence_list.append(Evidence(
                content=r["content"],
                citation=citation,
                relevance=1.0 - r.get("distance", 0.5) # Approx conversion
            ))
            
        return evidence_list

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to tracking."""
        self.hypotheses.append(hypothesis)

    def add_conflict(self, conflict: Conflict) -> None:
        """Add a detected conflict."""
        self.conflicts.append(conflict)

    def get_open_conflicts(self) -> list[Conflict]:
        """Get unresolved conflicts."""
        return [c for c in self.conflicts if c.status == "open"]

    def get_confirmed_hypotheses(self) -> list[Hypothesis]:
        """Get high-confidence hypotheses."""
        return [h for h in self.hypotheses if h.confidence > 0.8]
```

### Phase 2: Integrate into Simple Mode

**File:** `src/orchestrators/simple.py`

```python
# Add to __init__
from src.services.research_memory import ResearchMemory

class Orchestrator:
    def __init__(self, ...):
        ...
        self._memory: ResearchMemory | None = None

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        # Initialize memory for this run
        self._memory = ResearchMemory(query=query)

        # In search phase:
        new_ids = await self._memory.store_evidence(search_results.evidence)

        # In judge phase:
        relevant = await self._memory.get_relevant_evidence(n=30)
        # ... existing judge logic, but now with memory context

        # Track hypotheses from judge assessment
        for h in assessment.details.drug_candidates:
            self._memory.add_hypothesis(Hypothesis(
                id=h,
                statement=f"{h} identified as candidate",
                status="proposed",
                confidence=assessment.confidence,
            ))
```

### Phase 3: Integrate into Advanced Mode

**File:** `src/orchestrators/advanced.py`

```python
# Same pattern - inject ResearchMemory
# Agents read/write to shared memory instead of chat history
```

### Phase 4: Remove God Mode from UI

**File:** `src/app.py`

```python
# Before
mode = gr.Radio(
    choices=["simple", "magentic", "god"],
    ...
)

# After
mode = gr.Radio(
    choices=["simple", "magentic"],
    ...
)
# Memory is always enabled, not a mode choice
```

**File:** `src/orchestrators/factory.py`

```python
# Remove "god" and "langgraph" mode handling
# Keep LangGraphOrchestrator code for reference/future use
```

---

## 4. What Stays, What Goes

| Component | Action |
|-----------|--------|
| `src/agents/graph/state.py` | ✅ KEEP - Hypothesis/Conflict models |
| `src/agents/graph/nodes.py` | ⚠️ EXTRACT - Move memory logic to service |
| `src/agents/graph/workflow.py` | 📦 ARCHIVE - LangGraph routing (optional) |
| `src/orchestrators/langgraph_orchestrator.py` | 📦 ARCHIVE - Not needed if memory integrated |
| `src/services/research_memory.py` | ✨ NEW - Shared memory service |

---

## 5. Files to Modify

| File | Change |
|------|--------|
| `src/services/research_memory.py` | NEW - Extract from nodes.py |
| `src/orchestrators/simple.py` | Add memory integration |
| `src/orchestrators/advanced.py` | Add memory integration |
| `src/orchestrators/factory.py` | Remove "god" mode |
| `src/app.py` | Remove God Mode from dropdown |
| `tests/unit/services/test_research_memory.py` | NEW - Test memory service |

---

## 6. Acceptance Criteria

- [ ] `ResearchMemory` service extracted and tested
- [ ] Simple mode uses `ResearchMemory` for evidence storage
- [ ] Simple mode tracks hypotheses from judge assessments
- [ ] Advanced mode uses `ResearchMemory` (shared state)
- [ ] "God Mode" removed from UI
- [ ] All existing tests pass
- [ ] New tests for memory integration

---

## 7. Why This is the Right Pattern

```
Iterative Development:

1. Build in isolation    ✅ (SPEC_07 - God Mode)
   - Test without breaking existing code
   - Verify the concept works

2. Ship isolated feature ✅ (PR #72)
   - Get it into main
   - Real users can test it

3. Integrate into stack  🔜 (This spec)
   - Wire into existing modes
   - Remove scaffolding

4. Clean up              🔜
   - Delete God Mode UI
   - Archive LangGraph orchestrator
```

**You shipped the hard part. Now it's just plumbing.**

---

## 8. Time Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Extract memory service | 2 hours |
| Phase 2: Simple mode integration | 2 hours |
| Phase 3: Advanced mode integration | 2 hours |
| Phase 4: UI cleanup | 30 mins |
| Testing | 1 hour |
| **Total** | **~8 hours** |

---

## 9. References

- SPEC_07: LangGraph Memory Architecture (implemented)
- PR #72: God Mode implementation
- Issue #73: Architectural refactor tracking
