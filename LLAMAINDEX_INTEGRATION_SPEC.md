# LlamaIndex RAG Integration Specification

**Version:** 1.0.0
**Date:** 2025-11-30
**Author:** Claude (DeepBoner Singularity Initiative)
**Status:** IMPLEMENTATION READY

## Executive Summary

This specification details the integration of LlamaIndex RAG into DeepBoner's embedding infrastructure following SOLID principles, DRY patterns, and Gang of Four design patterns. The goal is to wire the orphaned `LlamaIndexRAGService` into the system via a tiered service selection mechanism.

---

## Architecture Overview

### Current State (Problem)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ResearchMemory ──────────────► EmbeddingService (always)       │
│       │                              │                           │
│       │                              ├── sentence-transformers   │
│       │                              ├── ChromaDB (in-memory)    │
│       │                              └── NO persistence          │
│       │                                                          │
│       │                                                          │
│  LlamaIndexRAGService ──────────► ORPHANED (never called)       │
│       │                              │                           │
│       │                              ├── OpenAI embeddings       │
│       │                              ├── ChromaDB (persistent)   │
│       │                              └── LlamaIndex RAG          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Target State (Solution)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TARGET ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ResearchMemory ──────────────► get_embedding_service()         │
│       │                              │                           │
│       │                              ▼                           │
│       │                    ┌─────────────────────┐               │
│       │                    │  Service Selection  │               │
│       │                    │  (Strategy Pattern) │               │
│       │                    └─────────────────────┘               │
│       │                         │           │                    │
│       │              ┌──────────┘           └──────────┐         │
│       │              ▼                                 ▼         │
│       │    ┌─────────────────┐           ┌───────────────────┐  │
│       │    │  EmbeddingService│          │LlamaIndexRAGService│  │
│       │    │  (Free Tier)     │          │(Premium Tier)      │  │
│       │    ├─────────────────┤           ├───────────────────┤  │
│       │    │ sentence-trans.  │          │ OpenAI embeddings  │  │
│       │    │ In-memory        │          │ Persistent storage │  │
│       │    │ No API key req.  │          │ Requires OPENAI_KEY│  │
│       │    └─────────────────┘           └───────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  EmbeddingServiceProtocol ◄──── Common Interface (Protocol)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns Applied

### 1. Strategy Pattern (Gang of Four)
**Purpose:** Allow interchangeable embedding services at runtime.

```python
# EmbeddingServiceProtocol defines the interface
# EmbeddingService and LlamaIndexRAGService are concrete strategies
# get_embedding_service() is the context that selects the strategy
```

### 2. Protocol Pattern (Structural Typing)
**Purpose:** Define interface without inheritance using Python's `typing.Protocol`.

```python
from typing import Protocol, Any
from src.utils.models import Evidence

class EmbeddingServiceProtocol(Protocol):
    """Duck-typed interface for embedding services."""

    async def add_evidence(self, evidence_id: str, content: str,
                          metadata: dict[str, Any]) -> None: ...
    async def search_similar(self, query: str,
                            n_results: int = 5) -> list[dict[str, Any]]: ...
    async def deduplicate(self, evidence: list[Evidence]) -> list[Evidence]: ...
```

### 3. Factory Method Pattern
**Purpose:** Encapsulate service creation logic.

```python
def get_embedding_service() -> EmbeddingServiceProtocol:
    """Factory method that returns the best available service."""
    if settings.has_openai_key:
        return _create_llamaindex_service()
    return _create_local_service()
```

### 4. Adapter Pattern
**Purpose:** Make LlamaIndexRAGService async-compatible with the protocol.

```python
# Wrap sync methods with async wrappers using run_in_executor
async def add_evidence(self, evidence_id: str, content: str,
                      metadata: dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, self._sync_add_evidence,
                               evidence_id, content, metadata)
```

### 5. Dependency Injection
**Purpose:** Allow ResearchMemory to receive any compatible embedding service.

```python
class ResearchMemory:
    def __init__(self, query: str,
                 embedding_service: EmbeddingServiceProtocol | None = None):
        self._embedding_service = embedding_service or get_embedding_service()
```

---

## SOLID Principles Applied

### Single Responsibility Principle (SRP)
- `EmbeddingService`: Handles local embeddings only
- `LlamaIndexRAGService`: Handles OpenAI embeddings + persistence only
- `service_loader`: Handles service selection only
- `EmbeddingServiceProtocol`: Defines interface only

### Open/Closed Principle (OCP)
- New embedding services can be added without modifying existing code
- Just implement `EmbeddingServiceProtocol` and register in `service_loader`

### Liskov Substitution Principle (LSP)
- Both `EmbeddingService` and `LlamaIndexRAGService` are substitutable
- They implement identical async interfaces

### Interface Segregation Principle (ISP)
- Protocol includes only methods needed by ResearchMemory
- No "fat interface" with unused methods

### Dependency Inversion Principle (DIP)
- ResearchMemory depends on `EmbeddingServiceProtocol` (abstraction)
- Not on concrete `EmbeddingService` or `LlamaIndexRAGService`

---

## DRY Principle Applied

### Before (Violation)
```python
# In EmbeddingService
await self.add_evidence(ev_id, content, {
    "source": ev.citation.source,
    "title": ev.citation.title,
    ...
})

# In LlamaIndexRAGService - DUPLICATE metadata building
doc = Document(text=ev.content, metadata={
    "source": evidence.citation.source,
    "title": evidence.citation.title,
    ...
})
```

### After (DRY)
```python
# In utils/models.py
class Evidence:
    def to_metadata(self) -> dict[str, Any]:
        """Convert to storage metadata format."""
        return {
            "source": self.citation.source,
            "title": self.citation.title,
            "date": self.citation.date,
            "authors": ",".join(self.citation.authors or []),
            "url": self.citation.url,
        }
```

---

## Implementation Files

### File 1: `src/services/embedding_protocol.py` (NEW)

```python
"""Protocol definition for embedding services.

This module defines the common interface that all embedding services must implement.
Using Protocol (PEP 544) for structural subtyping - no inheritance required.
"""

from typing import Any, Protocol

from src.utils.models import Evidence


class EmbeddingServiceProtocol(Protocol):
    """Common interface for embedding services.

    Both EmbeddingService (local/free) and LlamaIndexRAGService (OpenAI/premium)
    implement this interface, allowing seamless swapping via get_embedding_service().

    Design Pattern: Strategy Pattern (Gang of Four)
    - Each implementation is a concrete strategy
    - Protocol defines the strategy interface
    - service_loader selects the appropriate strategy at runtime
    """

    async def add_evidence(
        self, evidence_id: str, content: str, metadata: dict[str, Any]
    ) -> None:
        """Store evidence with embeddings.

        Args:
            evidence_id: Unique identifier (typically URL)
            content: Text content to embed
            metadata: Additional metadata for retrieval
        """
        ...

    async def search_similar(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Search for semantically similar content.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of dicts with keys: id, content, metadata, distance
        """
        ...

    async def deduplicate(
        self, evidence: list[Evidence], threshold: float = 0.9
    ) -> list[Evidence]:
        """Remove duplicate evidence based on semantic similarity.

        Args:
            evidence: List of evidence items to deduplicate
            threshold: Similarity threshold (0.9 = 90% similar is duplicate)

        Returns:
            List of unique evidence items
        """
        ...
```

### File 2: `src/utils/service_loader.py` (MODIFIED)

```python
"""Service loader utility for safe, lazy loading of optional services.

This module handles the import and initialization of services that may
have missing optional dependencies (like Modal or Sentence Transformers),
preventing the application from crashing if they are not available.

Design Patterns:
- Factory Method: get_embedding_service() creates appropriate service
- Strategy Pattern: Selects between EmbeddingService and LlamaIndexRAGService
"""

from typing import TYPE_CHECKING

import structlog

from src.utils.config import settings

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol
    from src.services.embeddings import EmbeddingService
    from src.services.llamaindex_rag import LlamaIndexRAGService
    from src.services.statistical_analyzer import StatisticalAnalyzer

logger = structlog.get_logger()


def get_embedding_service() -> "EmbeddingServiceProtocol":
    """Get the best available embedding service.

    Strategy selection (ordered by preference):
    1. LlamaIndexRAGService if OPENAI_API_KEY present (better quality + persistence)
    2. EmbeddingService (free, local, in-memory) as fallback

    Design Pattern: Factory Method + Strategy Pattern
    - Factory Method: Creates service instance
    - Strategy Pattern: Selects between implementations at runtime

    Returns:
        EmbeddingServiceProtocol: Either LlamaIndexRAGService or EmbeddingService

    Raises:
        ImportError: If no embedding service dependencies are available
    """
    # Try premium tier first (OpenAI + persistence)
    if settings.has_openai_key:
        try:
            from src.services.llamaindex_rag import get_rag_service

            service = get_rag_service()
            logger.info(
                "Using LlamaIndex RAG service",
                tier="premium",
                persistence="enabled",
                embeddings="openai",
            )
            return service
        except ImportError as e:
            logger.info(
                "LlamaIndex deps not installed, falling back to local embeddings",
                missing=str(e),
            )
        except Exception as e:
            logger.warning(
                "LlamaIndex service failed to initialize, falling back",
                error=str(e),
                error_type=type(e).__name__,
            )

    # Fallback to free tier (local embeddings, in-memory)
    try:
        from src.services.embeddings import get_embedding_service as get_local_service

        service = get_local_service()
        logger.info(
            "Using local embedding service",
            tier="free",
            persistence="disabled",
            embeddings="sentence-transformers",
        )
        return service
    except ImportError as e:
        logger.error(
            "No embedding service available",
            error=str(e),
        )
        raise ImportError(
            "No embedding service available. Install either:\n"
            "  - uv sync --extra embeddings (for local embeddings)\n"
            "  - uv sync --extra modal (for LlamaIndex with OpenAI)"
        ) from e


def get_embedding_service_if_available() -> "EmbeddingServiceProtocol | None":
    """
    Safely attempt to load and initialize an embedding service.

    Returns:
        EmbeddingServiceProtocol instance if dependencies are met, else None.
    """
    try:
        return get_embedding_service()
    except ImportError as e:
        logger.info(
            "Embedding service not available (optional dependencies missing)",
            missing_dependency=str(e),
        )
    except Exception as e:
        logger.warning(
            "Embedding service initialization failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
        )
    return None


def get_analyzer_if_available() -> "StatisticalAnalyzer | None":
    """
    Safely attempt to load and initialize the StatisticalAnalyzer.

    Returns:
        StatisticalAnalyzer instance if Modal is available, else None.
    """
    try:
        from src.services.statistical_analyzer import get_statistical_analyzer

        analyzer = get_statistical_analyzer()
        logger.info("StatisticalAnalyzer initialized successfully")
        return analyzer
    except ImportError as e:
        logger.info(
            "StatisticalAnalyzer not available (Modal dependencies missing)",
            missing_dependency=str(e),
        )
    except Exception as e:
        logger.warning(
            "StatisticalAnalyzer initialization failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
        )
    return None
```

### File 3: `src/services/llamaindex_rag.py` (MODIFIED - add async wrappers)

Add these methods to `LlamaIndexRAGService` class:

```python
# Add to imports at top
import asyncio

# Add these async wrapper methods to the class

async def add_evidence(
    self, evidence_id: str, content: str, metadata: dict[str, Any]
) -> None:
    """Async wrapper for adding evidence (Protocol-compatible).

    Converts the sync ingest_evidence pattern to the async protocol interface.
    Uses run_in_executor to avoid blocking the event loop.
    """
    from src.utils.models import Citation, Evidence

    # Reconstruct Evidence from parts
    citation = Citation(
        source=metadata.get("source", "web"),
        title=metadata.get("title", "Unknown"),
        url=evidence_id,
        date=metadata.get("date", "Unknown"),
        authors=(metadata.get("authors", "") or "").split(",") if metadata.get("authors") else [],
    )
    evidence = Evidence(content=content, citation=citation)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, self.ingest_evidence, [evidence])

async def search_similar(
    self, query: str, n_results: int = 5
) -> list[dict[str, Any]]:
    """Async wrapper for retrieve (Protocol-compatible).

    Returns results in the same format as EmbeddingService.search_similar().
    """
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, self.retrieve, query, n_results)

    # Convert to EmbeddingService format for compatibility
    return [
        {
            "id": r.get("metadata", {}).get("url", ""),
            "content": r.get("text", ""),
            "metadata": r.get("metadata", {}),
            "distance": 1.0 - (r.get("score", 0.5) or 0.5),  # Convert score to distance
        }
        for r in results
    ]

async def deduplicate(
    self, evidence: list["Evidence"], threshold: float = 0.9
) -> list["Evidence"]:
    """Async wrapper for deduplication (Protocol-compatible).

    Uses retrieve() to check for existing similar content.
    Stores unique evidence and returns the deduplicated list.
    """
    unique = []

    for ev in evidence:
        try:
            # Check for similar existing content
            similar = await self.search_similar(ev.content, n_results=1)

            # Check similarity threshold
            # distance 0 = identical, higher = more different
            is_duplicate = similar and similar[0]["distance"] < (1 - threshold)

            if not is_duplicate:
                unique.append(ev)
                # Store the new evidence
                await self.add_evidence(
                    evidence_id=ev.citation.url,
                    content=ev.content,
                    metadata={
                        "source": ev.citation.source,
                        "title": ev.citation.title,
                        "date": ev.citation.date,
                        "authors": ",".join(ev.citation.authors or []),
                    },
                )
        except Exception as e:
            # Log but don't fail - better to have duplicates than lose data
            logger.warning(
                "Failed to process evidence in deduplicate",
                url=ev.citation.url,
                error=str(e),
            )
            unique.append(ev)

    return unique
```

### File 4: `src/services/research_memory.py` (MODIFIED)

```python
"""Shared research memory layer for all orchestration modes."""

from typing import TYPE_CHECKING, Any

import structlog

from src.agents.graph.state import Conflict, Hypothesis
from src.utils.models import Citation, Evidence

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol

logger = structlog.get_logger()


class ResearchMemory:
    """Shared cognitive state for research workflows.

    This is the memory layer that ALL modes use.
    It mimics the LangGraph state management but for manual orchestration.

    Design Pattern: Dependency Injection
    - Receives embedding service via constructor
    - Uses service_loader.get_embedding_service() as default
    - Allows testing with mock services
    """

    def __init__(
        self,
        query: str,
        embedding_service: "EmbeddingServiceProtocol | None" = None
    ):
        """Initialize ResearchMemory with a query and optional embedding service.

        Args:
            query: The research query to track evidence for.
            embedding_service: Service for semantic search and deduplication.
                             Uses get_embedding_service() if not provided.
        """
        self.query = query
        self.hypotheses: list[Hypothesis] = []
        self.conflicts: list[Conflict] = []
        self.evidence_ids: list[str] = []
        self._evidence_cache: dict[str, Evidence] = {}
        self.iteration_count: int = 0

        # Lazy import to avoid circular dependencies
        if embedding_service is None:
            from src.utils.service_loader import get_embedding_service
            self._embedding_service = get_embedding_service()
        else:
            self._embedding_service = embedding_service

    # ... rest of the class remains the same ...
```

### File 5: `tests/unit/services/test_service_loader.py` (NEW)

```python
"""Tests for service loader embedding service selection."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetEmbeddingService:
    """Tests for get_embedding_service() tiered selection."""

    def test_uses_llamaindex_when_openai_key_present(self, monkeypatch):
        """Should return LlamaIndexRAGService when OPENAI_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")

        # Reset settings singleton to pick up new env var
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True

            # Mock LlamaIndex service
            mock_rag_service = MagicMock()
            with patch(
                "src.utils.service_loader.get_rag_service",
                return_value=mock_rag_service
            ):
                from src.utils.service_loader import get_embedding_service

                service = get_embedding_service()

                # Should be the LlamaIndex service
                assert service is mock_rag_service

    def test_falls_back_to_local_when_no_openai_key(self, monkeypatch):
        """Should return EmbeddingService when no OpenAI key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            # Mock local service
            mock_local_service = MagicMock()
            with patch(
                "src.services.embeddings.get_embedding_service",
                return_value=mock_local_service
            ):
                from src.utils.service_loader import get_embedding_service

                service = get_embedding_service()

                # Should be the local service
                assert service is mock_local_service

    def test_falls_back_when_llamaindex_import_fails(self, monkeypatch):
        """Should fallback to local if LlamaIndex deps missing."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True

            # LlamaIndex import fails
            def raise_import_error(*args, **kwargs):
                raise ImportError("llama_index not installed")

            mock_local_service = MagicMock()

            with patch.dict(
                "sys.modules",
                {"src.services.llamaindex_rag": None}
            ):
                with patch(
                    "src.services.embeddings.get_embedding_service",
                    return_value=mock_local_service
                ):
                    from src.utils.service_loader import get_embedding_service

                    # Should fallback gracefully
                    service = get_embedding_service()
                    assert service is mock_local_service

    def test_raises_when_no_embedding_service_available(self, monkeypatch):
        """Should raise ImportError when no embedding service can be loaded."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            # Both imports fail
            with patch.dict(
                "sys.modules",
                {
                    "src.services.llamaindex_rag": None,
                    "src.services.embeddings": None,
                }
            ):
                from src.utils.service_loader import get_embedding_service

                with pytest.raises(ImportError) as exc_info:
                    get_embedding_service()

                assert "No embedding service available" in str(exc_info.value)


class TestGetEmbeddingServiceIfAvailable:
    """Tests for get_embedding_service_if_available() safe wrapper."""

    def test_returns_none_when_no_service_available(self, monkeypatch):
        """Should return None instead of raising when no service available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch(
                "src.utils.service_loader.get_embedding_service",
                side_effect=ImportError("no deps")
            ):
                from src.utils.service_loader import get_embedding_service_if_available

                result = get_embedding_service_if_available()

                assert result is None

    def test_returns_service_when_available(self, monkeypatch):
        """Should return the service when available."""
        mock_service = MagicMock()

        with patch(
            "src.utils.service_loader.get_embedding_service",
            return_value=mock_service
        ):
            from src.utils.service_loader import get_embedding_service_if_available

            result = get_embedding_service_if_available()

            assert result is mock_service
```

### File 6: `tests/unit/services/test_llamaindex_rag_protocol.py` (NEW)

```python
"""Tests for LlamaIndexRAGService protocol compliance."""

from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

# Skip if LlamaIndex dependencies not installed
pytest.importorskip("llama_index")
pytest.importorskip("chromadb")


class TestLlamaIndexProtocolCompliance:
    """Verify LlamaIndexRAGService implements EmbeddingServiceProtocol."""

    @pytest.fixture
    def mock_openai_key(self, monkeypatch):
        """Provide a mock OpenAI key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")

    @pytest.fixture
    def mock_llamaindex_deps(self):
        """Mock all LlamaIndex dependencies."""
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_collection.return_value = mock_collection
            mock_chroma.return_value.create_collection.return_value = mock_collection

            with patch("llama_index.core.VectorStoreIndex") as mock_index:
                with patch("llama_index.core.Settings"):
                    with patch("llama_index.embeddings.openai.OpenAIEmbedding"):
                        with patch("llama_index.llms.openai.OpenAI"):
                            with patch("llama_index.vector_stores.chroma.ChromaVectorStore"):
                                yield {
                                    "chroma": mock_chroma,
                                    "collection": mock_collection,
                                    "index": mock_index,
                                }

    @pytest.mark.asyncio
    async def test_add_evidence_is_async(self, mock_openai_key, mock_llamaindex_deps):
        """add_evidence should be an async method."""
        from src.services.llamaindex_rag import LlamaIndexRAGService

        service = LlamaIndexRAGService()

        # Should be callable as async
        result = service.add_evidence("id", "content", {"source": "pubmed"})
        assert asyncio.iscoroutine(result)
        await result  # Clean up coroutine

    @pytest.mark.asyncio
    async def test_search_similar_is_async(self, mock_openai_key, mock_llamaindex_deps):
        """search_similar should be an async method."""
        from src.services.llamaindex_rag import LlamaIndexRAGService

        service = LlamaIndexRAGService()

        # Mock retrieve to avoid actual API call
        service.retrieve = MagicMock(return_value=[])

        result = service.search_similar("query", n_results=5)
        assert asyncio.iscoroutine(result)
        results = await result
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_deduplicate_is_async(self, mock_openai_key, mock_llamaindex_deps):
        """deduplicate should be an async method."""
        from src.services.llamaindex_rag import LlamaIndexRAGService
        from src.utils.models import Citation, Evidence

        service = LlamaIndexRAGService()

        # Mock search_similar
        service.search_similar = AsyncMock(return_value=[])
        service.add_evidence = AsyncMock()

        evidence = [
            Evidence(
                content="test",
                citation=Citation(source="pubmed", url="u1", title="t1", date="2024"),
            )
        ]

        result = service.deduplicate(evidence)
        assert asyncio.iscoroutine(result)
        unique = await result
        assert len(unique) == 1

    @pytest.mark.asyncio
    async def test_search_similar_returns_correct_format(
        self, mock_openai_key, mock_llamaindex_deps
    ):
        """search_similar should return EmbeddingService-compatible format."""
        from src.services.llamaindex_rag import LlamaIndexRAGService

        service = LlamaIndexRAGService()

        # Mock retrieve to return LlamaIndex format
        service.retrieve = MagicMock(return_value=[
            {
                "text": "some content",
                "score": 0.9,
                "metadata": {
                    "source": "pubmed",
                    "title": "Test",
                    "url": "http://example.com",
                },
            }
        ])

        results = await service.search_similar("query")

        assert len(results) == 1
        result = results[0]

        # Verify correct format
        assert "id" in result
        assert "content" in result
        assert "metadata" in result
        assert "distance" in result

        # Distance should be 1 - score
        assert result["distance"] == pytest.approx(0.1, abs=0.01)
```

---

## Bug Inventory (P0-P3)

### P0 - Critical (Must Fix)

**BUG-001: LlamaIndexRAGService not async-compatible**
- **Location:** `src/services/llamaindex_rag.py`
- **Issue:** All methods are sync, but ResearchMemory expects async
- **Fix:** Add async wrappers using `run_in_executor()`
- **Status:** PLANNED (this spec)

### P1 - High (Should Fix)

**BUG-002: ResearchMemory always creates new EmbeddingService**
- **Location:** `src/services/research_memory.py:37`
- **Issue:** `EmbeddingService()` called directly, bypassing service selection
- **Fix:** Use `get_embedding_service()` instead
- **Status:** PLANNED (this spec)

**BUG-003: Duplicate metadata construction logic**
- **Location:** `embeddings.py:156-161`, `llamaindex_rag.py:128-134`
- **Issue:** Same metadata dict built in multiple places (DRY violation)
- **Fix:** Add `Evidence.to_metadata()` method
- **Status:** OPTIONAL (nice-to-have)

### P2 - Medium (Could Fix)

**BUG-004: LlamaIndex score-to-distance conversion unclear**
- **Location:** `llamaindex_rag.py` (new code)
- **Issue:** LlamaIndex uses similarity scores (higher = better), EmbeddingService uses distance (lower = better)
- **Fix:** Document and test conversion: `distance = 1 - score`
- **Status:** PLANNED (this spec)

**BUG-005: No type hints for EmbeddingServiceProtocol in ResearchMemory**
- **Location:** `src/services/research_memory.py`
- **Issue:** `embedding_service` parameter typed as `EmbeddingService | None`
- **Fix:** Type as `EmbeddingServiceProtocol | None`
- **Status:** PLANNED (this spec)

### P3 - Low (Nice to Have)

**BUG-006: Singleton pattern for LlamaIndex service not implemented**
- **Location:** `src/services/llamaindex_rag.py`
- **Issue:** Each call to `get_rag_service()` creates new instance
- **Fix:** Add module-level singleton like `_shared_model` in `embeddings.py`
- **Status:** DEFERRED (not critical for hackathon)

**BUG-007: Missing integration test for tiered service selection**
- **Location:** `tests/integration/`
- **Issue:** No test verifies actual service switching with real keys
- **Fix:** Add integration test with conditional skip based on env
- **Status:** DEFERRED

---

## Implementation Order (TDD)

### Phase 1: Tests First (Red)
1. Create `tests/unit/services/test_service_loader.py`
2. Create `tests/unit/services/test_llamaindex_rag_protocol.py`
3. Run tests - all should fail (no implementation yet)

### Phase 2: Protocol (Green - Part 1)
1. Create `src/services/embedding_protocol.py`
2. Verify type checking passes

### Phase 3: LlamaIndex Async (Green - Part 2)
1. Add async wrappers to `src/services/llamaindex_rag.py`
2. Run protocol tests - should pass

### Phase 4: Service Loader (Green - Part 3)
1. Update `src/utils/service_loader.py`
2. Run service loader tests - should pass

### Phase 5: ResearchMemory (Green - Part 4)
1. Update `src/services/research_memory.py`
2. Run existing tests - all should pass

### Phase 6: Integration (Refactor)
1. Run `make check`
2. Fix any type errors or lint issues
3. Commit with clear message

---

## Acceptance Criteria

- [ ] `get_embedding_service()` returns `LlamaIndexRAGService` when `OPENAI_API_KEY` present
- [ ] Falls back to `EmbeddingService` when no OpenAI key
- [ ] Both services have compatible async interfaces (Protocol compliance)
- [ ] Persistence works (evidence survives restart with OpenAI key)
- [ ] All existing tests pass
- [ ] New tests for service selection
- [ ] `make check` passes (lint + typecheck + test)
- [ ] No regression in Gradio app functionality

---

## Sources & References

### LlamaIndex Best Practices 2025
- [LlamaIndex Production RAG Guide](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)
- [LlamaIndex + ChromaDB Integration](https://docs.trychroma.com/integrations/frameworks/llamaindex)
- [LlamaIndex Embeddings Documentation](https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/)

### Design Patterns
- Gang of Four: Strategy Pattern for service selection
- Python Protocol (PEP 544) for structural typing
- Factory Method for service creation

### SOLID Principles
- Single Responsibility: Each service has one job
- Open/Closed: New services don't require changes to existing code
- Liskov Substitution: Services are interchangeable
- Interface Segregation: Protocol has minimal methods
- Dependency Inversion: Depend on Protocol, not concrete classes

---

## Appendix: Full File Listing

After implementation, the following files will be modified or created:

| File | Status | Purpose |
|------|--------|---------|
| `src/services/embedding_protocol.py` | NEW | Protocol interface definition |
| `src/utils/service_loader.py` | MODIFIED | Add `get_embedding_service()` |
| `src/services/llamaindex_rag.py` | MODIFIED | Add async wrapper methods |
| `src/services/research_memory.py` | MODIFIED | Use service loader |
| `tests/unit/services/test_service_loader.py` | NEW | Service selection tests |
| `tests/unit/services/test_llamaindex_rag_protocol.py` | NEW | Protocol compliance tests |
