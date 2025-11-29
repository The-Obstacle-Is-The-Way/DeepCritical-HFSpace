"""Shared research memory layer for all orchestration modes."""

from typing import Any

import structlog

from src.agents.graph.state import Conflict, Hypothesis
from src.services.embeddings import EmbeddingService
from src.utils.models import Citation, Evidence

logger = structlog.get_logger()


class ResearchMemory:
    """Shared cognitive state for research workflows.

    This is the memory layer that ALL modes use.
    It mimics the LangGraph state management but for manual orchestration.
    """

    def __init__(self, query: str, embedding_service: EmbeddingService | None = None):
        """Initialize ResearchMemory with a query and optional embedding service.

        Args:
            query: The research query to track evidence for.
            embedding_service: Service for semantic search and deduplication.
                             Creates a new instance if not provided.
        """
        self.query = query
        self.hypotheses: list[Hypothesis] = []
        self.conflicts: list[Conflict] = []
        self.evidence_ids: list[str] = []
        self._evidence_cache: dict[str, Evidence] = {}
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
