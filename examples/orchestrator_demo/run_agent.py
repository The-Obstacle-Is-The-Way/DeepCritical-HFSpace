#!/usr/bin/env python3
"""
Demo: DeepBoner Agent Loop (Search + Judge + Orchestrator).

This script demonstrates the REAL Phase 4 orchestration:
- REAL Iterative Search (PubMed + ClinicalTrials + Europe PMC)
- REAL Evidence Evaluation (LLM Judge)
- REAL Orchestration Loop
- REAL Final Synthesis

NO MOCKS. REAL API CALLS.

Usage:
    uv run python examples/orchestrator_demo/run_agent.py "testosterone libido"
    uv run python examples/orchestrator_demo/run_agent.py "sildenafil erectile dysfunction" \
        --iterations 5

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import argparse
import asyncio
import os
import sys

from src.agent_factory.judges import JudgeHandler
from src.orchestrators import Orchestrator
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.models import OrchestratorConfig

MAX_ITERATIONS = 10


async def main() -> None:
    """Run the REAL agent demo."""
    parser = argparse.ArgumentParser(
        description="DeepBoner Agent Demo - REAL, No Mocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This demo runs the REAL search-judge-synthesize loop:
  1. REAL search: PubMed + ClinicalTrials + Europe PMC queries
  2. REAL judge: Actual LLM assessing evidence quality
  3. REAL loop: Actual iterative refinement based on LLM decisions
  4. REAL synthesis: Actual research summary generation

Examples:
    uv run python examples/orchestrator_demo/run_agent.py "testosterone libido"
    uv run python examples/orchestrator_demo/run_agent.py "flibanserin HSDD" --iterations 5
        """,
    )
    parser.add_argument("query", help="Research query (e.g., 'testosterone libido')")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations (default: 3)")
    args = parser.parse_args()

    if not 1 <= args.iterations <= MAX_ITERATIONS:
        print(f"Error: iterations must be between 1 and {MAX_ITERATIONS}")
        sys.exit(1)

    # Fail fast: require API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("=" * 60)
        print("ERROR: This demo requires a real LLM.")
        print()
        print("Set one of the following in your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print()
        print("This is a REAL demo. No mocks. No fake data.")
        print("=" * 60)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("DeepBoner Agent Demo (REAL)")
    print(f"Query: {args.query}")
    print(f"Max Iterations: {args.iterations}")
    print("Mode: REAL (All live API calls)")
    print(f"{'=' * 60}\n")

    # Setup REAL components
    search_handler = SearchHandler(
        tools=[PubMedTool(), ClinicalTrialsTool(), EuropePMCTool()], timeout=30.0
    )
    judge_handler = JudgeHandler()  # REAL LLM judge

    config = OrchestratorConfig(max_iterations=args.iterations)
    orchestrator = Orchestrator(
        search_handler=search_handler, judge_handler=judge_handler, config=config
    )

    # Run the REAL loop
    try:
        async for event in orchestrator.run(args.query):
            # Print event with icon (remove markdown bold for CLI)
            print(event.to_markdown().replace("**", ""))

            # Show search results count
            if event.type == "search_complete" and event.data:
                print(f"   -> Found {event.data.get('new_count', 0)} new items")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    print("\n" + "=" * 60)
    print("Demo complete! Everything was REAL:")
    print("  - Real PubMed + ClinicalTrials + Europe PMC searches")
    print("  - Real LLM judge decisions")
    print("  - Real iterative refinement")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
