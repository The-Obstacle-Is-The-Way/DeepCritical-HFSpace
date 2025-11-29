# NEXT_TASK: Wire LlamaIndex RAG Service into Simple Mode

**Priority:** P1 - Infrastructure
**GitHub Issues:** Addresses #64 (persistence) and #54 (wire in LlamaIndex)
**Difficulty:** Medium
**Estimated Changes:** 3-4 files

## Problem

We have two embedding services that are NOT connected:

1. `src/services/embeddings.py` - Used everywhere (free, in-memory, no persistence)
2. `src/services/llamaindex_rag.py` - Never used (better embeddings, persistence, RAG)

The LlamaIndex service provides significant value but is orphaned code.

## Solution: Tiered Service Selection

Use the existing `service_loader.py` pattern to select the right service:

```python
# When NO OpenAI key: Use free local embeddings (current behavior)
# When OpenAI key present: Upgrade to LlamaIndex (persistence + better quality)
```

## Implementation Steps

### Step 1: Add service selection in `src/utils/service_loader.py`

```python
def get_embedding_service() -> "EmbeddingService | LlamaIndexRAGService":
    """Get the best available embedding service.

    Returns LlamaIndexRAGService if OpenAI key available (better quality + persistence).
    Falls back to EmbeddingService (free, in-memory) otherwise.
    """
    if settings.openai_api_key:
        try:
            from src.services.llamaindex_rag import get_rag_service
            return get_rag_service()
        except ImportError:
            pass  # LlamaIndex deps not installed, fallback

    from src.services.embeddings import EmbeddingService
    return EmbeddingService()
```

### Step 2: Create a unified interface (Protocol)

Both services need compatible methods. Create `src/services/embedding_protocol.py`:

```python
from typing import Protocol, Any
from src.utils.models import Evidence

class EmbeddingServiceProtocol(Protocol):
    """Common interface for embedding services."""

    async def add_evidence(self, evidence_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Store evidence with embeddings."""
        ...

    async def search_similar(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Search for similar content."""
        ...

    async def deduplicate(self, evidence: list[Evidence]) -> list[Evidence]:
        """Remove duplicate evidence."""
        ...
```

### Step 3: Make LlamaIndexRAGService async-compatible

Current `llamaindex_rag.py` methods are sync. Wrap them:

```python
async def add_evidence(self, evidence_id: str, content: str, metadata: dict[str, Any]) -> None:
    """Async wrapper for ingest."""
    loop = asyncio.get_running_loop()
    evidence = Evidence(content=content, citation=Citation(...metadata))
    await loop.run_in_executor(None, self.ingest_evidence, [evidence])
```

### Step 4: Update ResearchMemory to use the service loader

In `src/services/research_memory.py`:

```python
from src.utils.service_loader import get_embedding_service

class ResearchMemory:
    def __init__(self, query: str, embedding_service: EmbeddingServiceProtocol | None = None):
        self._embedding_service = embedding_service or get_embedding_service()
```

### Step 5: Add tests

```python
# tests/unit/services/test_service_loader.py
def test_uses_llamaindex_when_openai_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service = get_embedding_service()
    assert isinstance(service, LlamaIndexRAGService)

def test_falls_back_to_local_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = get_embedding_service()
    assert isinstance(service, EmbeddingService)
```

## Benefits After Implementation

| Feature | Free Tier | Premium Tier (OpenAI key) |
|---------|-----------|---------------------------|
| Embeddings | Local (sentence-transformers) | OpenAI (text-embedding-3-small) |
| Persistence | In-memory (lost on restart) | Disk (ChromaDB PersistentClient) |
| Quality | Good | Better |
| Cost | Free | API costs |
| Knowledge accumulation | No | Yes |

## Files to Modify

1. `src/utils/service_loader.py` - Add `get_embedding_service()`
2. `src/services/llamaindex_rag.py` - Add async wrappers, match interface
3. `src/services/research_memory.py` - Use service loader
4. `tests/unit/services/test_service_loader.py` - Add tests

## Acceptance Criteria

- [ ] `get_embedding_service()` returns LlamaIndex when OpenAI key present
- [ ] Falls back to local EmbeddingService when no key
- [ ] Both services have compatible async interfaces
- [ ] Persistence works (evidence survives restart with OpenAI key)
- [ ] All existing tests pass
- [ ] New tests for service selection

## Related Issues

- #64 - feat: Add persistence to EmbeddingService (this solves it via LlamaIndex)
- #54 - tech-debt: LlamaIndex RAG is dead code (this wires it in)

## Notes for AI Agent

- Run `make check` before committing
- The service_loader.py pattern already exists for Modal - follow that pattern
- LlamaIndex requires `uv sync --extra modal` for deps
- Test with and without OPENAI_API_KEY set
