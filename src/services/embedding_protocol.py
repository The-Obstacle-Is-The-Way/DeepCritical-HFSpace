"""Protocol definition for embedding services.

This module defines the common interface that all embedding services must implement.
Using Protocol (PEP 544) for structural subtyping - no inheritance required.

Design Pattern: Strategy Pattern (Gang of Four)
- Each implementation (EmbeddingService, LlamaIndexRAGService) is a concrete strategy
- Protocol defines the strategy interface
- service_loader selects the appropriate strategy at runtime

SOLID Principles:
- Interface Segregation: Protocol includes only methods needed by consumers
- Dependency Inversion: Consumers depend on Protocol (abstraction), not concrete classes
- Liskov Substitution: All implementations are interchangeable
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.utils.models import Evidence


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Common interface for embedding services.

    Both EmbeddingService (local/free) and LlamaIndexRAGService (OpenAI/premium)
    implement this interface, allowing seamless swapping via get_embedding_service().

    All methods are async to avoid blocking the event loop during:
    - Embedding computation (CPU-bound with local models)
    - Vector store operations (I/O-bound with persistent storage)
    - API calls (network I/O with OpenAI embeddings)

    Example:
        ```python
        from src.utils.service_loader import get_embedding_service

        # Get best available service (LlamaIndex if OpenAI key, else local)
        service = get_embedding_service()

        # Use via protocol interface
        await service.add_evidence("id", "content", {"source": "pubmed"})
        results = await service.search_similar("query", n_results=5)
        unique = await service.deduplicate(evidence_list)
        ```
    """

    async def add_evidence(
        self, evidence_id: str, content: str, metadata: dict[str, Any]
    ) -> None:
        """Store evidence with embeddings.

        Args:
            evidence_id: Unique identifier (typically URL)
            content: Text content to embed and store
            metadata: Additional metadata for retrieval filtering
                Expected keys: source, title, date, authors, url
        """
        ...

    async def search_similar(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Search for semantically similar content.

        Args:
            query: Search query text
            n_results: Maximum number of results to return

        Returns:
            List of dicts with keys:
            - id: Evidence identifier
            - content: Original text content
            - metadata: Stored metadata
            - distance: Semantic distance (0 = identical, higher = less similar)
        """
        ...

    async def deduplicate(
        self, evidence: list["Evidence"], threshold: float = 0.9
    ) -> list["Evidence"]:
        """Remove duplicate evidence based on semantic similarity.

        Uses the embedding service to check if new evidence is similar to
        existing stored evidence. Unique evidence is stored automatically.

        Args:
            evidence: List of evidence items to deduplicate
            threshold: Similarity threshold (0.9 = 90% similar is duplicate)
                ChromaDB cosine distance interpretation:
                - 0 = identical vectors
                - 2 = opposite vectors
                Duplicate if: distance < (1 - threshold)

        Returns:
            List of unique evidence items (duplicates removed)
        """
        ...
