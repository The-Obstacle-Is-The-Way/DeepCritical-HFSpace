#!/usr/bin/env python3
"""
Demo: Full Stack DeepCritical Agent (Phases 1-8).

This script demonstrates the COMPLETE REAL drug repurposing research pipeline:
- Phase 2: REAL Search (PubMed + Web API calls)
- Phase 6: REAL Embeddings (sentence-transformers + ChromaDB)
- Phase 7: REAL Hypothesis (LLM mechanistic reasoning)
- Phase 3: REAL Judge (LLM evidence assessment)
- Phase 8: REAL Report (LLM structured scientific report)

NO MOCKS. NO FAKE DATA. REAL SCIENCE.

Usage:
    uv run python examples/full_stack_demo/run_full.py "metformin Alzheimer's"
    uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" -i 3

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from src.utils.models import Evidence


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step: int, name: str) -> None:
    """Print a step indicator."""
    print(f"\n[Step {step}] {name}")
    print("-" * 50)


_MAX_DISPLAY_LEN = 600


def _print_truncated(text: str) -> None:
    """Print text, truncating if too long."""
    if len(text) > _MAX_DISPLAY_LEN:
        print(text[:_MAX_DISPLAY_LEN] + "\n... [truncated for display]")
    else:
        print(text)


async def _run_search_iteration(
    query: str,
    iteration: int,
    evidence_store: dict[str, Any],
    all_evidence: list[Evidence],
    search_handler: Any,
    embedding_service: Any,
) -> list[Evidence]:
    """Run a single search iteration with deduplication."""
    search_queries = [query]
    if evidence_store.get("hypotheses"):
        for h in evidence_store["hypotheses"][-2:]:
            search_queries.extend(h.search_suggestions[:1])

    for q in search_queries[:2]:
        result = await search_handler.execute(q, max_results_per_tool=5)
        print(f"  '{q}' -> {result.total_found} results")
        new_unique = await embedding_service.deduplicate(result.evidence)
        print(f"  After dedup: {len(new_unique)} unique")
        all_evidence.extend(new_unique)

    evidence_store["current"] = all_evidence
    evidence_store["iteration_count"] = iteration
    return all_evidence


async def _handle_judge_step(
    judge_handler: Any, query: str, all_evidence: list[Evidence], evidence_store: dict[str, Any]
) -> tuple[bool, str]:
    """Handle the judge assessment step. Returns (should_stop, next_query)."""
    print("\n[Judge] Assessing evidence quality (REAL LLM)...")
    assessment = await judge_handler.assess(query, all_evidence)
    print(f"  Mechanism Score: {assessment.details.mechanism_score}/10")
    print(f"  Clinical Score:  {assessment.details.clinical_evidence_score}/10")
    print(f"  Confidence:      {assessment.confidence:.0%}")
    print(f"  Recommendation:  {assessment.recommendation.upper()}")

    if assessment.recommendation == "synthesize":
        print("\n[Judge] Evidence sufficient! Proceeding to report generation...")
        evidence_store["last_assessment"] = assessment.details.model_dump()
        return True, query

    next_queries = assessment.next_search_queries[:2] if assessment.next_search_queries else []
    if next_queries:
        print(f"\n[Judge] Need more evidence. Next queries: {next_queries}")
        return False, next_queries[0]

    print(
        "\n[Judge] Need more evidence but no suggested queries. " "Continuing with original query."
    )
    return False, query


async def run_full_demo(query: str, max_iterations: int) -> None:
    """Run the REAL full stack pipeline."""
    print_header("DeepCritical Full Stack Demo (REAL)")
    print(f"Query: {query}")
    print(f"Max iterations: {max_iterations}")
    print("Mode: REAL (All live API calls - no mocks)\n")

    # Import real components
    from src.agent_factory.judges import JudgeHandler
    from src.agents.hypothesis_agent import HypothesisAgent
    from src.agents.report_agent import ReportAgent
    from src.services.embeddings import EmbeddingService
    from src.tools.pubmed import PubMedTool
    from src.tools.search_handler import SearchHandler
    from src.tools.websearch import WebTool

    # Initialize REAL services
    print("[Init] Loading embedding model...")
    embedding_service = EmbeddingService()
    search_handler = SearchHandler(tools=[PubMedTool(), WebTool()], timeout=30.0)
    judge_handler = JudgeHandler()

    # Shared evidence store
    evidence_store: dict[str, Any] = {"current": [], "hypotheses": [], "iteration_count": 0}
    all_evidence: list[Evidence] = []

    for iteration in range(1, max_iterations + 1):
        print_step(iteration, f"ITERATION {iteration}/{max_iterations}")

        # Step 1: REAL Search
        print("\n[Search] Querying PubMed and Web (REAL API calls)...")
        all_evidence = await _run_search_iteration(
            query, iteration, evidence_store, all_evidence, search_handler, embedding_service
        )

        if not all_evidence:
            print("\nNo evidence found. Try a different query.")
            return

        # Step 2: REAL Hypothesis generation (first iteration only)
        if iteration == 1:
            print("\n[Hypothesis] Generating mechanistic hypotheses (REAL LLM)...")
            hypothesis_agent = HypothesisAgent(evidence_store, embedding_service)
            hyp_response = await hypothesis_agent.run(query)
            _print_truncated(hyp_response.messages[0].text)

        # Step 3: REAL Judge
        should_stop, query = await _handle_judge_step(
            judge_handler, query, all_evidence, evidence_store
        )
        if should_stop:
            break

    # Step 4: REAL Report generation
    print_step(iteration + 1, "REPORT GENERATION (REAL LLM)")
    report_agent = ReportAgent(evidence_store, embedding_service)
    report_response = await report_agent.run(query)

    print("\n" + "=" * 70)
    print("  FINAL RESEARCH REPORT")
    print("=" * 70)
    print(report_response.messages[0].text)


async def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="DeepCritical Full Stack Demo - REAL, No Mocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This demo runs the COMPLETE pipeline with REAL API calls:
  1. REAL search: Actual PubMed + DuckDuckGo queries
  2. REAL embeddings: Actual sentence-transformers model
  3. REAL hypothesis: Actual LLM generating mechanistic chains
  4. REAL judge: Actual LLM assessing evidence quality
  5. REAL report: Actual LLM generating structured report

Examples:
    uv run python examples/full_stack_demo/run_full.py "metformin Alzheimer's"
    uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" -i 3
    uv run python examples/full_stack_demo/run_full.py "aspirin cancer prevention"
        """,
    )
    parser.add_argument(
        "query",
        help="Research query (e.g., 'metformin Alzheimer's disease')",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=2,
        help="Max search iterations (default: 2)",
    )

    args = parser.parse_args()

    if args.iterations < 1:
        print("Error: iterations must be at least 1")
        sys.exit(1)

    # Fail fast: require API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("=" * 70)
        print("ERROR: This demo requires a real LLM.")
        print()
        print("Set one of the following in your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print()
        print("This is a REAL demo. No mocks. No fake data.")
        print("=" * 70)
        sys.exit(1)

    await run_full_demo(args.query, args.iterations)

    print("\n" + "=" * 70)
    print("  DeepCritical Full Stack Demo Complete!")
    print("  ")
    print("  Everything you just saw was REAL:")
    print("    - Real PubMed/Web searches")
    print("    - Real embedding computations")
    print("    - Real LLM reasoning")
    print("    - Real scientific report")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
