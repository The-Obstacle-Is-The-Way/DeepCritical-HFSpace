#!/usr/bin/env python3
"""
Demo: Hypothesis Generation (Phase 7).

This script demonstrates the REAL hypothesis generation pipeline:
1. REAL search: PubMed + Web (actual API calls)
2. REAL embeddings: Semantic deduplication
3. REAL LLM: Mechanistic hypothesis generation

Usage:
    # Requires OPENAI_API_KEY or ANTHROPIC_API_KEY
    uv run python examples/hypothesis_demo/run_hypothesis.py "metformin Alzheimer's"
    uv run python examples/hypothesis_demo/run_hypothesis.py "sildenafil heart failure"
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from src.agents.hypothesis_agent import HypothesisAgent
from src.services.embeddings import EmbeddingService
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.tools.websearch import WebTool


async def run_hypothesis_demo(query: str) -> None:
    """Run the REAL hypothesis generation pipeline."""
    try:
        print(f"\n{'='*60}")
        print("DeepCritical Hypothesis Agent Demo (Phase 7)")
        print(f"Query: {query}")
        print("Mode: REAL (Live API calls)")
        print(f"{'='*60}\n")

        # Step 1: REAL Search
        print("[Step 1] Searching PubMed + Web...")
        search_handler = SearchHandler(tools=[PubMedTool(), WebTool()], timeout=30.0)
        result = await search_handler.execute(query, max_results_per_tool=5)

        print(f"  Found {result.total_found} results from {result.sources_searched}")
        if result.errors:
            print(f"  Warnings: {result.errors}")

        if not result.evidence:
            print("\nNo evidence found. Try a different query.")
            return

        # Step 2: REAL Embeddings - Deduplicate
        print("\n[Step 2] Semantic deduplication...")
        embedding_service = EmbeddingService()
        unique_evidence = await embedding_service.deduplicate(result.evidence, threshold=0.85)
        print(f"  {len(result.evidence)} -> {len(unique_evidence)} unique papers")

        # Show what we found
        print("\n[Evidence collected]")
        max_title_len = 50
        for i, e in enumerate(unique_evidence[:5], 1):
            raw_title = e.citation.title
            if len(raw_title) > max_title_len:
                title = raw_title[:max_title_len] + "..."
            else:
                title = raw_title
            print(f"  {i}. [{e.citation.source.upper()}] {title}")

        # Step 3: REAL LLM - Generate hypotheses
        print("\n[Step 3] Generating mechanistic hypotheses (LLM)...")
        evidence_store: dict[str, Any] = {"current": unique_evidence, "hypotheses": []}
        agent = HypothesisAgent(evidence_store, embedding_service)

        print("-" * 60)
        response = await agent.run(query)
        print(response.messages[0].text)
        print("-" * 60)

        # Show stored hypotheses
        hypotheses = evidence_store.get("hypotheses", [])
        print(f"\n{len(hypotheses)} hypotheses stored")

        if hypotheses:
            print("\nGenerated search queries for further investigation:")
            for h in hypotheses:
                queries = h.to_search_queries()
                print(f"  {h.drug} -> {h.target}:")
                for q in queries[:3]:
                    print(f"    - {q}")

    except Exception as e:
        print(f"\n❌ Error during hypothesis generation: {e}")
        raise


async def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Hypothesis Generation Demo (REAL - No Mocks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python examples/hypothesis_demo/run_hypothesis.py "metformin Alzheimer's"
    uv run python examples/hypothesis_demo/run_hypothesis.py "sildenafil heart failure"
    uv run python examples/hypothesis_demo/run_hypothesis.py "aspirin cancer prevention"
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="metformin Alzheimer's disease",
        help="Research query",
    )
    args = parser.parse_args()

    # Fail fast: require API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("=" * 60)
        print("ERROR: This demo requires a real LLM.")
        print()
        print("Set one of the following in your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print()
        print("This is a REAL demo, not a mock. No fake data.")
        print("=" * 60)
        sys.exit(1)

    await run_hypothesis_demo(args.query)

    print("\n" + "=" * 60)
    print("Demo complete! This was a REAL pipeline:")
    print("  1. REAL search: Actual PubMed + Web API calls")
    print("  2. REAL embeddings: Actual sentence-transformers")
    print("  3. REAL LLM: Actual hypothesis generation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
