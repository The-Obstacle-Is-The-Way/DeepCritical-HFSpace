#!/usr/bin/env python3
"""Demo: Modal-powered statistical analysis.

This script uses StatisticalAnalyzer directly (NO agent_framework dependency).

Usage:
    uv run python examples/modal_demo/run_analysis.py "metformin alzheimer"
"""

import argparse
import asyncio
import os
import sys

from src.services.statistical_analyzer import get_statistical_analyzer
from src.tools.pubmed import PubMedTool
from src.utils.config import settings


async def main() -> None:
    """Run the Modal analysis demo."""
    parser = argparse.ArgumentParser(description="Modal Analysis Demo")
    parser.add_argument("query", help="Research query")
    args = parser.parse_args()

    if not settings.modal_available:
        print("Error: Modal credentials not configured.")
        sys.exit(1)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: No LLM API key found.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("DeepCritical Modal Analysis Demo")
    print(f"Query: {args.query}")
    print(f"{ '=' * 60}\n")

    # Step 1: Gather Evidence
    print("Step 1: Gathering evidence from PubMed...")
    pubmed = PubMedTool()
    evidence = await pubmed.search(args.query, max_results=5)
    print(f"  Found {len(evidence)} papers\n")

    # Step 2: Run Modal Analysis
    print("Step 2: Running statistical analysis in Modal sandbox...")
    analyzer = get_statistical_analyzer()
    result = await analyzer.analyze(query=args.query, evidence=evidence)

    # Step 3: Display Results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nVerdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.0%}")
    print("\nKey Findings:")
    for finding in result.key_findings:
        print(f"  - {finding}")

    print("\n[Demo Complete - Code executed in Modal, not locally]")


if __name__ == "__main__":
    asyncio.run(main())
