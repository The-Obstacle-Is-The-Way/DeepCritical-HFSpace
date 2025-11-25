"""Orchestrator - the agent loop connecting Search and Judge."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Protocol

import structlog

from src.utils.models import (
    AgentEvent,
    Evidence,
    JudgeAssessment,
    OrchestratorConfig,
    SearchResult,
)

logger = structlog.get_logger()


class SearchHandlerProtocol(Protocol):
    """Protocol for search handler."""

    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult: ...


class JudgeHandlerProtocol(Protocol):
    """Protocol for judge handler."""

    async def assess(self, question: str, evidence: list[Evidence]) -> JudgeAssessment: ...


class Orchestrator:
    """
    The agent orchestrator - runs the Search -> Judge -> Loop cycle.

    This is a generator-based design that yields events for real-time UI updates.
    """

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        judge_handler: JudgeHandlerProtocol,
        config: OrchestratorConfig | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            search_handler: Handler for executing searches
            judge_handler: Handler for assessing evidence
            config: Optional configuration (uses defaults if not provided)
        """
        self.search = search_handler
        self.judge = judge_handler
        self.config = config or OrchestratorConfig()
        self.history: list[dict[str, Any]] = []

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
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

                # Deduplicate evidence by URL
                seen_urls = {e.citation.url for e in all_evidence}
                unique_new = [e for e in new_evidence if e.citation.url not in seen_urls]
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
