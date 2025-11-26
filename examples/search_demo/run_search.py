#!/usr/bin/env python3
"""
Demo: Search for drug repurposing evidence.

This script demonstrates Phase 2 functionality:
- PubMed search (biomedical literature)
- SearchHandler (parallel scatter-gather orchestration)

Usage:
    # From project root:
    uv run python examples/search_demo/run_search.py

    # With custom query:
    uv run python examples/search_demo/run_search.py "metformin cancer"

Requirements:
    - Optional: NCBI_API_KEY in .env for higher PubMed rate limits
"""

import asyncio
import sys

from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler


async def main(query: str) -> None:
    """Run search demo with the given query."""
    print(f"\n{'='*60}")
    print("DeepCritical Search Demo")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Initialize tools
    pubmed = PubMedTool()
    handler = SearchHandler(tools=[pubmed], timeout=30.0)

    # Execute search
    print("Searching PubMed in parallel...")
    result = await handler.execute(query, max_results_per_tool=5)

    # Display results
    print(f"\n{'='*60}")
    print(f"Results: {result.total_found} pieces of evidence")
    print(f"Sources: {', '.join(result.sources_searched)}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print(f"{'='*60}\n")

    for i, evidence in enumerate(result.evidence, 1):
        print(f"[{i}] {evidence.citation.source.upper()}: {evidence.citation.title[:80]}...")
        print(f"    URL: {evidence.citation.url}")
        print(f"    Content: {evidence.content[:150]}...")
        print()


if __name__ == "__main__":
    # Default query or use command line arg
    default_query = "metformin Alzheimer's disease drug repurposing"
    query = sys.argv[1] if len(sys.argv) > 1 else default_query

    asyncio.run(main(query))
