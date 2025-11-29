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
from typing import TYPE_CHECKING, Any, ClassVar

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

    # Termination thresholds (code-enforced, not LLM-decided)
    TERMINATION_CRITERIA: ClassVar[dict[str, float]] = {
        "min_combined_score": 12.0,  # mechanism + clinical >= 12
        "min_score_with_volume": 10.0,  # >= 10 if 50+ sources
        "min_evidence_for_volume": 50.0,  # Priority 3: evidence count threshold
        "late_iteration_threshold": 8.0,  # >= 8 in iterations 8+
        "max_evidence_threshold": 100.0,  # Force synthesis with 100+ sources
        "emergency_iteration": 8.0,  # Last 2 iterations = emergency mode
        "min_confidence": 0.5,  # Minimum confidence for emergency synthesis
        "min_evidence_for_emergency": 30.0,  # Priority 6: min evidence for emergency
    }

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
        """Lazy initialization of StatisticalAnalyzer."""
        if self._analyzer is None:
            from src.utils.service_loader import get_analyzer_if_available

            self._analyzer = get_analyzer_if_available()
            if self._analyzer is None:
                self._enable_analysis = False
        return self._analyzer

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

    def _should_synthesize(
        self,
        assessment: JudgeAssessment,
        iteration: int,
        max_iterations: int,
        evidence_count: int,
    ) -> tuple[bool, str]:
        """
        Code-enforced synthesis decision.

        Returns (should_synthesize, reason).
        """
        combined_score = (
            assessment.details.mechanism_score + assessment.details.clinical_evidence_score
        )
        has_drug_candidates = len(assessment.details.drug_candidates) > 0
        confidence = assessment.confidence

        # Priority 1: LLM explicitly says sufficient with good scores
        if assessment.sufficient and assessment.recommendation == "synthesize":
            if combined_score >= 10:
                return True, "judge_approved"

        # Priority 2: High scores with drug candidates
        if (
            combined_score >= self.TERMINATION_CRITERIA["min_combined_score"]
            and has_drug_candidates
        ):
            return True, "high_scores_with_candidates"

        # Priority 3: Good scores with high evidence volume
        if (
            combined_score >= self.TERMINATION_CRITERIA["min_score_with_volume"]
            and evidence_count >= self.TERMINATION_CRITERIA["min_evidence_for_volume"]
        ):
            return True, "good_scores_high_volume"

        # Priority 4: Late iteration with acceptable scores (diminishing returns)
        is_late_iteration = iteration >= max_iterations - 2
        if (
            is_late_iteration
            and combined_score >= self.TERMINATION_CRITERIA["late_iteration_threshold"]
        ):
            return True, "late_iteration_acceptable"

        # Priority 5: Very high evidence count (enough to synthesize something)
        if evidence_count >= self.TERMINATION_CRITERIA["max_evidence_threshold"]:
            return True, "max_evidence_reached"

        # Priority 6: Emergency synthesis (avoid garbage output)
        if (
            is_late_iteration
            and evidence_count >= self.TERMINATION_CRITERIA["min_evidence_for_emergency"]
            and confidence >= self.TERMINATION_CRITERIA["min_confidence"]
        ):
            return True, "emergency_synthesis"

        return False, "continue_searching"

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:  # noqa: PLR0915
        """
        Run the agent loop for a query.

        Yields AgentEvent objects for each step, allowing real-time UI updates.

        Args:
            query: The user's research question

        Yields:
            AgentEvent objects for each step of the process
        """
        # Import here to avoid circular deps if any
        from src.agents.graph.state import Hypothesis
        from src.services.research_memory import ResearchMemory

        logger.info("Starting orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research for: {query}",
            iteration=0,
        )

        # Initialize Shared Memory
        # We keep 'all_evidence' for local tracking/reporting, but use Memory for intelligence
        memory = ResearchMemory(query=query)
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

                # === MEMORY INTEGRATION: Store and Deduplicate ===
                # ResearchMemory handles semantic deduplication and persistence
                # It returns IDs of actual NEW evidence
                new_ids = await memory.store_evidence(new_evidence)

                # Filter new_evidence to only keep what was actually new (based on IDs)
                # Note: This assumes IDs are URLs, which match Citation.url
                unique_new = [e for e in new_evidence if e.citation.url in new_ids]

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
                message=f"Evaluating evidence (Memory: {len(memory.evidence_ids)} docs)...",
                iteration=iteration,
            )

            try:
                # Retrieve RELEVANT evidence from memory for the judge
                # This keeps the context window manageable and focused
                judge_context = await memory.get_relevant_evidence(n=30)

                # Fallback if memory is empty (shouldn't happen if search worked)
                if not judge_context and all_evidence:
                    judge_context = all_evidence[-30:]

                assessment = await self.judge.assess(
                    query, judge_context, iteration, self.config.max_iterations
                )

                # === MEMORY INTEGRATION: Track Hypotheses ===
                # Convert loose strings to structured Hypotheses
                for candidate in assessment.details.drug_candidates:
                    h = Hypothesis(
                        id=candidate.replace(" ", "_").lower(),
                        statement=f"{candidate} is a potential candidate for {query}",
                        status="proposed",
                        confidence=assessment.confidence,
                        reasoning=f" identified in iteration {iteration}",
                    )
                    memory.add_hypothesis(h)

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

                # === DECISION PHASE (Code-Enforced) ===
                should_synth, reason = self._should_synthesize(
                    assessment=assessment,
                    iteration=iteration,
                    max_iterations=self.config.max_iterations,
                    evidence_count=len(all_evidence),
                )

                logger.info(
                    "Synthesis decision",
                    should_synthesize=should_synth,
                    reason=reason,
                    iteration=iteration,
                    combined_score=assessment.details.mechanism_score
                    + assessment.details.clinical_evidence_score,
                    evidence_count=len(all_evidence),
                    confidence=assessment.confidence,
                )

                if should_synth:
                    # Log synthesis trigger reason for debugging
                    if reason != "judge_approved":
                        logger.info(f"Code-enforced synthesis triggered: {reason}")

                    # Optional Analysis Phase
                    async for event in self._run_analysis_phase(query, all_evidence, iteration):
                        yield event

                    yield AgentEvent(
                        type="synthesizing",
                        message=f"Evidence sufficient ({reason})! Preparing synthesis...",
                        iteration=iteration,
                    )

                    # Generate final response
                    # Use all gathered evidence for the final report
                    final_response = self._generate_synthesis(query, all_evidence, assessment)

                    yield AgentEvent(
                        type="complete",
                        message=final_response,
                        data={
                            "evidence_count": len(all_evidence),
                            "iterations": iteration,
                            "synthesis_reason": reason,
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
                            f"Gathering more evidence (scores: {assessment.details.mechanism_score}"
                            f"+{assessment.details.clinical_evidence_score}). "
                            f"Next: {', '.join(current_queries[:2])}..."
                        ),
                        data={"next_queries": current_queries, "reason": reason},
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
        Generate a REAL synthesis when max iterations reached.

        Even when forced to stop, we should provide:
        - Drug candidates (if any were found)
        - Key findings
        - Assessment scores
        - Actionable citations

        This is still better than a citation dump.
        """
        # Extract data from last assessment if available
        last_assessment = self.history[-1]["assessment"] if self.history else {}
        details = last_assessment.get("details", {})

        drug_candidates = details.get("drug_candidates", [])
        key_findings = details.get("key_findings", [])
        mechanism_score = details.get("mechanism_score", 0)
        clinical_score = details.get("clinical_evidence_score", 0)
        reasoning = last_assessment.get("reasoning", "Analysis incomplete due to iteration limit.")

        # Format drug candidates
        if drug_candidates:
            drug_list = "\n".join([f"- **{d}**" for d in drug_candidates[:5]])
        else:
            drug_list = (
                "- *No specific drug candidates identified in evidence*\n"
                "- *Try a more specific query or add an API key for better analysis*"
            )

        # Format key findings
        if key_findings:
            findings_list = "\n".join([f"- {f}" for f in key_findings[:5]])
        else:
            findings_list = (
                "- *Key findings require further analysis*\n"
                "- *See citations below for relevant sources*"
            )

        # Format citations (top 10)
        citations = "\n".join(
            [
                f"{i + 1}. [{e.citation.title}]({e.citation.url}) "
                f"({e.citation.source.upper()}, {e.citation.date})"
                for i, e in enumerate(evidence[:10])
            ]
        )

        combined_score = mechanism_score + clinical_score
        mech_strength = (
            "Strong" if mechanism_score >= 7 else "Moderate" if mechanism_score >= 4 else "Limited"
        )
        clin_strength = (
            "Strong" if clinical_score >= 7 else "Moderate" if clinical_score >= 4 else "Limited"
        )
        comb_strength = "Sufficient" if combined_score >= 12 else "Partial"

        return f"""## Drug Repurposing Analysis

### Research Question
{query}

### Status
Analysis based on {len(evidence)} sources across {len(self.history)} iterations.
Maximum iterations reached - results may be incomplete.

### Drug Candidates Identified
{drug_list}

### Key Findings
{findings_list}

### Evidence Quality Scores
| Criterion | Score | Interpretation |
|-----------|-------|----------------|
| Mechanism | {mechanism_score}/10 | {mech_strength} mechanistic evidence |
| Clinical | {clinical_score}/10 | {clin_strength} clinical support |
| Combined | {combined_score}/20 | {comb_strength} for synthesis |

### Analysis Summary
{reasoning}

### Top Citations ({len(evidence)} sources total)
{citations}

---
*For more complete analysis:*
- *Add an OpenAI or Anthropic API key for enhanced LLM analysis*
- *Try a more specific query (e.g., include drug names)*
- *Use Advanced mode for multi-agent research*
"""
