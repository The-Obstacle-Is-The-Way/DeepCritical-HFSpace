# Bug 005: Embedding Services Built But Not Wired to Default Orchestrator

**Date:** November 26, 2025
**Severity:** CRITICAL
**Status:** Open

## 1. The Problem

Two complete semantic search services exist but are **NOT USED** by the default orchestrator:

| Service | Location | Status |
| ------- | -------- | ------ |
| EmbeddingService | `src/services/embeddings.py` | BUILT, not wired to simple mode |
| LlamaIndexRAGService | `src/services/llamaindex_rag.py` | BUILT, not wired to simple mode |

## 2. Root Cause: Two Orchestrators

```
┌─────────────────────────────────────────────────────────────────┐
│ orchestrator.py (SIMPLE MODE - DEFAULT)                         │
│ - Basic search → judge → loop                                   │
│ - NO embeddings                                                 │
│ - NO semantic search                                            │
│ - Hand-rolled keyword matching                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ orchestrator_magentic.py (MAGENTIC MODE)                        │
│ - Multi-agent architecture                                      │
│ - USES EmbeddingService                                         │
│ - USES semantic search                                          │
│ - Requires agent-framework (optional dep)                       │
│ - OpenAI only                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**The UI defaults to simple mode**, which bypasses all the semantic search infrastructure.

## 3. What's Built (Not Wired)

### EmbeddingService (NO API KEY NEEDED)

```python
# src/services/embeddings.py
class EmbeddingService:
    async def embed(text) -> list[float]
    async def search_similar(query) -> list[dict]  # SEMANTIC SEARCH
    async def deduplicate(evidence) -> list        # DEDUPLICATION
```

- Uses local sentence-transformers
- ChromaDB vector store
- **Works without API keys**

### LlamaIndexRAGService

```python
# src/services/llamaindex_rag.py
class LlamaIndexRAGService:
    def ingest_evidence(evidence_list)
    def retrieve(query) -> list[dict]  # Semantic retrieval
    def query(query_str) -> str        # Synthesized response
```

## 4. Where Services ARE Used

```
src/orchestrator_magentic.py    ← Uses EmbeddingService
src/agents/search_agent.py      ← Uses EmbeddingService
src/agents/report_agent.py      ← Uses EmbeddingService
src/agents/hypothesis_agent.py  ← Uses EmbeddingService
src/agents/analysis_agent.py    ← Uses EmbeddingService
```

All in magentic mode agents, NOT in simple orchestrator.

## 5. The Fix Options

### Option A: Add Embeddings to Simple Orchestrator (RECOMMENDED)

Modify `src/orchestrator.py` to optionally use EmbeddingService:

```python
class Orchestrator:
    def __init__(self, ..., use_embeddings: bool = True):
        if use_embeddings:
            from src.services.embeddings import get_embedding_service
            self.embeddings = get_embedding_service()
        else:
            self.embeddings = None

    async def run(self, query):
        # ... search phase ...

        if self.embeddings:
            # Semantic ranking
            all_evidence = await self._rank_by_relevance(all_evidence, query)
            # Deduplication
            all_evidence = await self.embeddings.deduplicate(all_evidence)
```

### Option B: Make Magentic Mode Default

Change app.py to default to "magentic" mode when deps available.

### Option C: Merge Best of Both

Create a new orchestrator that:
- Has the simplicity of simple mode
- Uses embeddings for ranking/dedup
- Doesn't require agent-framework

## 6. Implementation Plan

### Phase 1: Wire EmbeddingService to Simple Orchestrator

1. Import EmbeddingService in orchestrator.py
2. Add semantic ranking after search
3. Add deduplication before judge
4. Test end-to-end

### Phase 2: Add Relevance to Evidence

1. Use embedding similarity as relevance score
2. Sort evidence by relevance
3. Only send top-K to judge

## 7. Files to Modify

```
src/orchestrator.py           ← Add embedding integration
src/orchestrator_factory.py   ← Pass embeddings flag
src/app.py                    ← Enable embeddings by default
```

## 8. Success Criteria

- [ ] Default mode uses semantic search
- [ ] Evidence ranked by relevance
- [ ] Duplicates removed
- [ ] No new API keys required (sentence-transformers is local)
- [ ] Magentic mode still works as before
