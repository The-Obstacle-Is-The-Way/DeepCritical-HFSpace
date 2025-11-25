# Phase 6 Implementation Spec: Embeddings & Semantic Search

**Goal**: Add vector search for semantic evidence retrieval.
**Philosophy**: "Find what you mean, not just what you type."
**Prerequisite**: Phase 5 complete (Magentic working)

---

## 1. Why Embeddings?

Current limitation: **Keyword-only search misses semantically related papers.**

Example problem:
- User searches: "metformin alzheimer"
- PubMed returns: Papers with exact keywords
- MISSED: Papers about "AMPK activation neuroprotection" (same mechanism, different words)

With embeddings:
- Embed the query AND all evidence
- Find semantically similar papers even without keyword match
- Deduplicate by meaning, not just URL

---

## 2. Architecture

### Current (Phase 5)
```
Query → SearchAgent → PubMed/Web (keyword) → Evidence
```

### Phase 6
```
Query → Embed(Query) → SearchAgent
                          ├── PubMed/Web (keyword) → Evidence
                          └── VectorDB (semantic) → Related Evidence
                                    ↑
                          Evidence → Embed → Store
```

### Shared Context Enhancement
```python
# Current
evidence_store = {"current": []}

# Phase 6
evidence_store = {
    "current": [],           # Raw evidence
    "embeddings": {},        # URL -> embedding vector
    "vector_index": None,    # ChromaDB collection
}
```

---

## 3. Technology Choice

### ChromaDB (Recommended)
- **Free**, open-source, local-first
- No API keys, no cloud dependency
- Supports sentence-transformers out of the box
- Perfect for hackathon (no infra setup)

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality)
- Or `BAAI/bge-small-en-v1.5` (better quality, still fast)

---

## 4. Implementation

### 4.1 Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
embeddings = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
]
```

### 4.2 Embedding Service (`src/services/embeddings.py`)

> **CRITICAL: Async Pattern Required**
>
> `sentence-transformers` is synchronous and CPU-bound. Running it directly in async code
> will **block the event loop**, freezing the UI and halting all concurrent operations.
>
> **Solution**: Use `asyncio.run_in_executor()` to offload to thread pool.
> This pattern already exists in `src/tools/websearch.py:28-34`.

```python
"""Embedding service for semantic search.

IMPORTANT: All public methods are async to avoid blocking the event loop.
The sentence-transformers model is CPU-bound, so we use run_in_executor().
"""
import asyncio
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Handles text embedding and vector storage.

    All embedding operations run in a thread pool to avoid blocking
    the async event loop. See src/tools/websearch.py for the pattern.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._client = chromadb.Client()  # In-memory for hackathon
        self._collection = self._client.create_collection(
            name="evidence",
            metadata={"hnsw:space": "cosine"}
        )

    # ─────────────────────────────────────────────────────────────────
    # Sync internal methods (run in thread pool)
    # ─────────────────────────────────────────────────────────────────

    def _sync_embed(self, text: str) -> List[float]:
        """Synchronous embedding - DO NOT call directly from async code."""
        return self._model.encode(text).tolist()

    def _sync_batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for efficiency - DO NOT call directly from async code."""
        return [e.tolist() for e in self._model.encode(texts)]

    # ─────────────────────────────────────────────────────────────────
    # Async public methods (safe for event loop)
    # ─────────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """Embed a single text (async-safe).

        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_embed, text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts (async-safe, more efficient)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_batch_embed, texts)

    async def add_evidence(self, evidence_id: str, content: str, metadata: dict) -> None:
        """Add evidence to vector store (async-safe)."""
        embedding = await self.embed(content)
        # ChromaDB operations are fast, but wrap for consistency
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[evidence_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[content]
            )
        )

    async def search_similar(self, query: str, n_results: int = 5) -> List[dict]:
        """Find semantically similar evidence (async-safe)."""
        query_embedding = await self.embed(query)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        )

        # Handle empty results gracefully
        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {"id": id, "content": doc, "metadata": meta, "distance": dist}
            for id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    async def deduplicate(self, new_evidence: List, threshold: float = 0.9) -> List:
        """Remove semantically duplicate evidence (async-safe)."""
        unique = []
        for evidence in new_evidence:
            similar = await self.search_similar(evidence.content, n_results=1)
            if not similar or similar[0]["distance"] > (1 - threshold):
                unique.append(evidence)
                await self.add_evidence(
                    evidence_id=evidence.citation.url,
                    content=evidence.content,
                    metadata={"source": evidence.citation.source}
                )
        return unique
```

### 4.3 Enhanced SearchAgent (`src/agents/search_agent.py`)

Update SearchAgent to use embeddings. **Note**: All embedding calls are `await`ed:

```python
class SearchAgent(BaseAgent):
    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        evidence_store: dict,
        embedding_service: EmbeddingService | None = None,  # NEW
    ):
        # ... existing init ...
        self._embeddings = embedding_service

    async def run(self, messages, *, thread=None, **kwargs) -> AgentRunResponse:
        # ... extract query ...

        # Execute keyword search
        result = await self._handler.execute(query, max_results_per_tool=10)

        # Semantic deduplication (NEW) - ALL CALLS ARE AWAITED
        if self._embeddings:
            # Deduplicate by semantic similarity (async-safe)
            unique_evidence = await self._embeddings.deduplicate(result.evidence)

            # Also search for semantically related evidence (async-safe)
            related = await self._embeddings.search_similar(query, n_results=5)

            # Merge related evidence not already in results
            existing_urls = {e.citation.url for e in unique_evidence}
            for item in related:
                if item["id"] not in existing_urls:
                    # Reconstruct Evidence from stored data
                    # ... merge logic ...

        # ... rest of method ...
```

### 4.4 Semantic Expansion in Orchestrator

The MagenticOrchestrator can use embeddings to expand queries:

```python
# In task instruction
task = f"""Research drug repurposing opportunities for: {query}

The system has semantic search enabled. When evidence is found:
1. Related concepts will be automatically surfaced
2. Duplicates are removed by meaning, not just URL
3. Use the surfaced related concepts to refine searches
"""
```

### 4.5 HuggingFace Spaces Deployment

> **⚠️ Important for HF Spaces**
>
> `sentence-transformers` downloads models (~500MB) to `~/.cache` on first use.
> HuggingFace Spaces have **ephemeral storage** - the cache is wiped on restart.
> This causes slow cold starts and bandwidth usage.

**Solution**: Pre-download the model in your Dockerfile:

```dockerfile
# In Dockerfile
FROM python:3.11-slim

# Set cache directory
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Pre-download the embedding model during build
RUN pip install sentence-transformers && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ... rest of Dockerfile
```

**Alternative**: Use environment variable to specify persistent path:

```yaml
# In HF Spaces settings or app.yaml
env:
  - name: HF_HOME
    value: /data/.cache  # Persistent volume
```

---

## 5. Directory Structure After Phase 6

```
src/
├── services/                   # NEW
│   ├── __init__.py
│   └── embeddings.py           # EmbeddingService
├── agents/
│   ├── search_agent.py         # Updated with embeddings
│   └── judge_agent.py
└── ...
```

---

## 6. Tests

### 6.1 Unit Tests (`tests/unit/services/test_embeddings.py`)

> **Note**: All tests are async since the EmbeddingService methods are async.

```python
"""Unit tests for EmbeddingService."""
import pytest
from src.services.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_returns_vector(self):
        """Embedding should return a float vector."""
        service = EmbeddingService()
        embedding = await service.embed("metformin diabetes")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_similar_texts_have_close_embeddings(self):
        """Semantically similar texts should have similar embeddings."""
        service = EmbeddingService()
        e1 = await service.embed("metformin treats diabetes")
        e2 = await service.embed("metformin is used for diabetes treatment")
        e3 = await service.embed("the weather is sunny today")

        # Cosine similarity helper
        from numpy import dot
        from numpy.linalg import norm
        cosine = lambda a, b: dot(a, b) / (norm(a) * norm(b))

        # Similar texts should be closer
        assert cosine(e1, e2) > cosine(e1, e3)

    @pytest.mark.asyncio
    async def test_batch_embed_efficient(self):
        """Batch embedding should be more efficient than individual calls."""
        service = EmbeddingService()
        texts = ["text one", "text two", "text three"]

        # Batch embed
        batch_results = await service.embed_batch(texts)
        assert len(batch_results) == 3
        assert all(isinstance(e, list) for e in batch_results)

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """Should be able to add evidence and search for similar."""
        service = EmbeddingService()
        await service.add_evidence(
            evidence_id="test1",
            content="Metformin activates AMPK pathway",
            metadata={"source": "pubmed"}
        )

        results = await service.search_similar("AMPK activation drugs", n_results=1)
        assert len(results) == 1
        assert "AMPK" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_similar_empty_collection(self):
        """Search on empty collection should return empty list, not error."""
        service = EmbeddingService()
        results = await service.search_similar("anything", n_results=5)
        assert results == []
```

---

## 7. Definition of Done

Phase 6 is **COMPLETE** when:

1. `EmbeddingService` implemented with ChromaDB
2. SearchAgent uses embeddings for deduplication
3. Semantic search surfaces related evidence
4. All unit tests pass
5. Integration test shows improved recall (finds related papers)

---

## 8. Value Delivered

| Before (Phase 5) | After (Phase 6) |
|------------------|-----------------|
| Keyword-only search | Semantic + keyword search |
| URL-based deduplication | Meaning-based deduplication |
| Miss related papers | Surface related concepts |
| Exact match required | Fuzzy semantic matching |

**Real example improvement:**
- Query: "metformin alzheimer"
- Before: Only papers mentioning both words
- After: Also finds "AMPK neuroprotection", "biguanide cognitive", etc.
