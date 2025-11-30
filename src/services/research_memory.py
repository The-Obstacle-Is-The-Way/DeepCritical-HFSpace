"""Shared research memory layer for all orchestration modes.

Design Pattern: Dependency Injection
- Receives embedding service via constructor
- Uses service_loader.get_embedding_service() as default (Strategy Pattern)
- Allows testing with mock services

SOLID Principles:
- Dependency Inversion: Depends on EmbeddingServiceProtocol, not concrete class
- Open/Closed: Works with any service implementing the protocol
"""

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

    The embedding service is selected via get_embedding_service(), which returns:
    - LlamaIndexRAGService (premium tier) if OPENAI_API_KEY is available
    - EmbeddingService (free tier) as fallback
    """

    def __init__(self, query: str, embedding_service: "EmbeddingServiceProtocol | None" = None):
        """Initialize ResearchMemory with a query and optional embedding service.

        Args:
            query: The research query to track evidence for.
            embedding_service: Service for semantic search and deduplication.
                             Uses get_embedding_service() if not provided,
                             which selects the best available service.
        """
        self.query = query
        self.hypotheses: list[Hypothesis] = []
        self.conflicts: list[Conflict] = []
        self.evidence_ids: list[str] = []
        self._evidence_cache: dict[str, Evidence] = {}
        self.iteration_count: int = 0

        # Use service loader for tiered service selection (Strategy Pattern)
        if embedding_service is None:
            from src.utils.service_loader import get_embedding_service

            self._embedding_service: EmbeddingServiceProtocol = get_embedding_service()
        else:
            self._embedding_service = embedding_service

    async def store_evidence(self, evidence: list[Evidence]) -> list[str]:
        """Store evidence and return new IDs (deduped)."""
        if not self._embedding_service:
            return []

        # Deduplicate and store (deduplicate() already calls add_evidence() internally)
        unique = await self._embedding_service.deduplicate(evidence)

        # Track IDs and cache (evidence already stored by deduplicate())
        new_ids = []
        for ev in unique:
            ev_id = ev.citation.url
            new_ids.append(ev_id)
            self._evidence_cache[ev_id] = ev

        self.evidence_ids.extend(new_ids)
        if new_ids:
            logger.info("Stored new evidence", count=len(new_ids))
        return new_ids

    def get_all_evidence(self) -> list[Evidence]:
        """Get all accumulated evidence objects."""
        return list(self._evidence_cache.values())

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
            source_raw = meta.get("source", "web")

            # Basic validation/fallback for source
            valid_sources = [
                "pubmed",
                "clinicaltrials",
                "europepmc",
                "preprint",
                "openalex",
                "web",
            ]
            source_name: Any = source_raw if source_raw in valid_sources else "web"

            citation = Citation(
                source=source_name,
                title=meta.get("title", "Unknown"),
                url=meta.get("url", r.get("id", "")),
                date=meta.get("date", "Unknown"),
                authors=authors,
            )

            evidence_list.append(
                Evidence(
                    content=r.get("content", ""),
                    citation=citation,
                    relevance=1.0 - r.get("distance", 0.5),  # Approx conversion
                )
            )

        return evidence_list

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to tracking."""
        self.hypotheses.append(hypothesis)
        logger.info("Added hypothesis", id=hypothesis.id, confidence=hypothesis.confidence)

    def add_conflict(self, conflict: Conflict) -> None:
        """Add a detected conflict."""
        self.conflicts.append(conflict)
        logger.info("Added conflict", id=conflict.id)

    def get_open_conflicts(self) -> list[Conflict]:
        """Get unresolved conflicts."""
        return [c for c in self.conflicts if c.status == "open"]

    def get_confirmed_hypotheses(self) -> list[Hypothesis]:
        """Get high-confidence hypotheses."""
        return [h for h in self.hypotheses if h.confidence > 0.8]
