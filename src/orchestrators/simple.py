"""Simple Orchestrator - the basic agent loop connecting Search and Judge.

This orchestrator uses a simple loop pattern with pydantic-ai for structured
LLM outputs. It works with free tier (HuggingFace Inference) or paid APIs
(OpenAI, Anthropic).

Design Pattern: Template Method - defines the skeleton of the search-judge loop
while allowing handlers to implement specific behaviors.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import structlog

from src.orchestrators.base import JudgeHandlerProtocol, SearchHandlerProtocol
from src.utils.config import settings
from src.utils.models import (
    AgentEvent,
    Evidence,
    JudgeAssessment,
    OrchestratorConfig,
    SearchResult,
)

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService
    from src.services.statistical_analyzer import StatisticalAnalyzer

logger = structlog.get_logger()


class Orchestrator:
    """
    The simple agent orchestrator - runs the Search -> Judge -> Loop cycle.

    This is a generator-based design that yields events for real-time UI updates.
    Uses pydantic-ai for structured LLM outputs without requiring the full
    Microsoft Agent Framework.
    """

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        judge_handler: JudgeHandlerProtocol,
        config: OrchestratorConfig | None = None,
        enable_analysis: bool = False,
        enable_embeddings: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            search_handler: Handler for executing searches
            judge_handler: Handler for assessing evidence
            config: Optional configuration (uses defaults if not provided)
            enable_analysis: Whether to perform statistical analysis (if Modal available)
            enable_embeddings: Whether to use semantic search for ranking/dedup
        """
        self.search = search_handler
        self.judge = judge_handler
        self.config = config or OrchestratorConfig()
        self.history: list[dict[str, Any]] = []
        self._enable_analysis = enable_analysis and settings.modal_available
        self._enable_embeddings = enable_embeddings

        # Lazy-load services (typed for IDE support)
        self._analyzer: StatisticalAnalyzer | None = None
        self._embeddings: EmbeddingService | None = None

    def _get_analyzer(self) -> StatisticalAnalyzer | None:
        """Lazy initialization of StatisticalAnalyzer.

        Note: This imports from src.services, NOT src.agents,
        so it works without the magentic optional dependency.

        Returns:
            StatisticalAnalyzer instance, or None if Modal is unavailable
        """
        if self._analyzer is None:
            try:
                from src.services.statistical_analyzer import get_statistical_analyzer

                self._analyzer = get_statistical_analyzer()
            except ImportError:
                logger.info("StatisticalAnalyzer not available (Modal dependencies missing)")
                self._enable_analysis = False
        return self._analyzer

    def _get_embeddings(self) -> EmbeddingService | None:
        """Lazy initialization of EmbeddingService.

        Uses local sentence-transformers - NO API key required.

        Returns:
            EmbeddingService instance, or None if unavailable
        """
        if self._embeddings is None and self._enable_embeddings:
            try:
                from src.services.embeddings import get_embedding_service

                self._embeddings = get_embedding_service()
                logger.info("Embedding service enabled for semantic ranking")
            except ImportError:
                logger.info("Embedding service not available (dependencies missing)")
                self._enable_embeddings = False
            except Exception as e:
                logger.warning(
                    "Embedding service initialization failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._enable_embeddings = False
        return self._embeddings

    async def _deduplicate_and_rank(self, evidence: list[Evidence], query: str) -> list[Evidence]:
        """Use embeddings to deduplicate and rank evidence by relevance."""
        embeddings = self._get_embeddings()
        if not embeddings or not evidence:
            return evidence

        try:
            # Deduplicate using semantic similarity
            unique_evidence: list[Evidence] = await embeddings.deduplicate(evidence, threshold=0.85)
            logger.info(
                "Deduplicated evidence",
                before=len(evidence),
                after=len(unique_evidence),
            )
            return unique_evidence
        except Exception as e:
            logger.warning("Deduplication failed, using original", error=str(e))
            return evidence

    async def _run_analysis_phase(
        self, query: str, evidence: list[Evidence], iteration: int
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the optional analysis phase."""
        if not self._enable_analysis:
            return

        yield AgentEvent(
            type="analyzing",
            message="Running statistical analysis in Modal sandbox...",
            data={},
            iteration=iteration,
        )

        try:
            analyzer = self._get_analyzer()
            if analyzer is None:
                logger.info("StatisticalAnalyzer not available, skipping analysis phase")
                return

            # Run Modal analysis (no agent_framework needed!)
            analysis_result = await analyzer.analyze(
                query=query,
                evidence=evidence,
                hypothesis=None,  # Could add hypothesis generation later
            )

            yield AgentEvent(
                type="analysis_complete",
                message=f"Analysis verdict: {analysis_result.verdict}",
                data=analysis_result.model_dump(),
                iteration=iteration,
            )

        except Exception as e:
            logger.error("Modal analysis failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Modal analysis failed: {e}",
                data={"error": str(e)},
                iteration=iteration,
            )

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:  # noqa: PLR0915
        """
        Run the agent loop for a query.

        Yields AgentEvent objects for each step, allowing real-time UI updates.

        Args:
            query: The user's research question

        Yields:
            AgentEvent objects for each step of the process
        """
        logger.info("Starting orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research for: {query}",
            iteration=0,
        )

        all_evidence: list[Evidence] = []
        current_queries = [query]
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info("Iteration", iteration=iteration, queries=current_queries)

            # === SEARCH PHASE ===
            yield AgentEvent(
                type="searching",
                message=f"Searching for: {', '.join(current_queries[:3])}...",
                iteration=iteration,
            )

            try:
                # Execute searches for all current queries
                search_tasks = [
                    self.search.execute(q, self.config.max_results_per_tool)
                    for q in current_queries[:3]  # Limit to 3 queries per iteration
                ]
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Collect evidence from successful searches
                new_evidence: list[Evidence] = []
                errors: list[str] = []

                for q, result in zip(current_queries[:3], search_results, strict=False):
                    if isinstance(result, Exception):
                        errors.append(f"Search for '{q}' failed: {result!s}")
                    elif isinstance(result, SearchResult):
                        new_evidence.extend(result.evidence)
                        errors.extend(result.errors)
                    else:
                        # Should not happen with return_exceptions=True but safe fallback
                        errors.append(f"Unknown result type for '{q}': {type(result)}")

                # Deduplicate evidence by URL (fast, basic)
                seen_urls = {e.citation.url for e in all_evidence}
                unique_new = [e for e in new_evidence if e.citation.url not in seen_urls]

                # BUG FIX: Only dedup NEW evidence, not all_evidence
                # Old evidence is already in the vector store - re-checking it
                # would mark items as duplicates of themselves (distance ≈ 0)
                if unique_new:
                    unique_new = await self._deduplicate_and_rank(unique_new, query)

                all_evidence.extend(unique_new)

                yield AgentEvent(
                    type="search_complete",
                    message=f"Found {len(unique_new)} new sources ({len(all_evidence)} total)",
                    data={
                        "new_count": len(unique_new),
                        "total_count": len(all_evidence),
                    },
                    iteration=iteration,
                )

                if errors:
                    logger.warning("Search errors", errors=errors)

            except Exception as e:
                logger.error("Search phase failed", error=str(e))
                yield AgentEvent(
                    type="error",
                    message=f"Search failed: {e!s}",
                    iteration=iteration,
                )
                continue

            # === JUDGE PHASE ===
            yield AgentEvent(
                type="judging",
                message=f"Evaluating {len(all_evidence)} sources...",
                iteration=iteration,
            )

            try:
                assessment = await self.judge.assess(query, all_evidence)

                yield AgentEvent(
                    type="judge_complete",
                    message=(
                        f"Assessment: {assessment.recommendation} "
                        f"(confidence: {assessment.confidence:.0%})"
                    ),
                    data={
                        "sufficient": assessment.sufficient,
                        "confidence": assessment.confidence,
                        "mechanism_score": assessment.details.mechanism_score,
                        "clinical_score": assessment.details.clinical_evidence_score,
                    },
                    iteration=iteration,
                )

                # Record this iteration in history
                self.history.append(
                    {
                        "iteration": iteration,
                        "queries": current_queries,
                        "evidence_count": len(all_evidence),
                        "assessment": assessment.model_dump(),
                    }
                )

                # === DECISION PHASE ===
                if assessment.sufficient and assessment.recommendation == "synthesize":
                    # Optional Analysis Phase
                    async for event in self._run_analysis_phase(query, all_evidence, iteration):
                        yield event

                    yield AgentEvent(
                        type="synthesizing",
                        message="Evidence sufficient! Preparing synthesis...",
                        iteration=iteration,
                    )

                    # Generate final response
                    final_response = self._generate_synthesis(query, all_evidence, assessment)

                    yield AgentEvent(
                        type="complete",
                        message=final_response,
                        data={
                            "evidence_count": len(all_evidence),
                            "iterations": iteration,
                            "drug_candidates": assessment.details.drug_candidates,
                            "key_findings": assessment.details.key_findings,
                        },
                        iteration=iteration,
                    )
                    return

                else:
                    # Need more evidence - prepare next queries
                    current_queries = assessment.next_search_queries or [
                        f"{query} mechanism of action",
                        f"{query} clinical evidence",
                    ]

                    yield AgentEvent(
                        type="looping",
                        message=(
                            f"Need more evidence. "
                            f"Next searches: {', '.join(current_queries[:2])}..."
                        ),
                        data={"next_queries": current_queries},
                        iteration=iteration,
                    )

            except Exception as e:
                logger.error("Judge phase failed", error=str(e))
                yield AgentEvent(
                    type="error",
                    message=f"Assessment failed: {e!s}",
                    iteration=iteration,
                )
                continue

        # Max iterations reached
        yield AgentEvent(
            type="complete",
            message=self._generate_partial_synthesis(query, all_evidence),
            data={
                "evidence_count": len(all_evidence),
                "iterations": iteration,
                "max_reached": True,
            },
            iteration=iteration,
        )

    def _generate_synthesis(
        self,
        query: str,
        evidence: list[Evidence],
        assessment: JudgeAssessment,
    ) -> str:
        """
        Generate the final synthesis response.

        Args:
            query: The original question
            evidence: All collected evidence
            assessment: The final assessment

        Returns:
            Formatted synthesis as markdown
        """
        drug_list = (
            "\n".join([f"- **{d}**" for d in assessment.details.drug_candidates])
            or "- No specific candidates identified"
        )
        findings_list = (
            "\n".join([f"- {f}" for f in assessment.details.key_findings]) or "- See evidence below"
        )

        citations = "\n".join(
            [
                f"{i + 1}. [{e.citation.title}]({e.citation.url}) "
                f"({e.citation.source.upper()}, {e.citation.date})"
                for i, e in enumerate(evidence[:10])  # Limit to 10 citations
            ]
        )

        return f"""## Drug Repurposing Analysis

### Question
{query}

### Drug Candidates
{drug_list}

### Key Findings
{findings_list}

### Assessment
- **Mechanism Score**: {assessment.details.mechanism_score}/10
- **Clinical Evidence Score**: {assessment.details.clinical_evidence_score}/10
- **Confidence**: {assessment.confidence:.0%}

### Reasoning
{assessment.reasoning}

### Citations ({len(evidence)} sources)
{citations}

---
*Analysis based on {len(evidence)} sources across {len(self.history)} iterations.*
"""

    def _generate_partial_synthesis(
        self,
        query: str,
        evidence: list[Evidence],
    ) -> str:
        """
        Generate a partial synthesis when max iterations reached.

        Args:
            query: The original question
            evidence: All collected evidence

        Returns:
            Formatted partial synthesis as markdown
        """
        citations = "\n".join(
            [
                f"{i + 1}. [{e.citation.title}]({e.citation.url}) ({e.citation.source.upper()})"
                for i, e in enumerate(evidence[:10])
            ]
        )

        return f"""## Partial Analysis (Max Iterations Reached)

### Question
{query}

### Status
Maximum search iterations reached. The evidence gathered may be incomplete.

### Evidence Collected
Found {len(evidence)} sources. Consider refining your query for more specific results.

### Citations
{citations}

---
*Consider searching with more specific terms or drug names.*
"""
