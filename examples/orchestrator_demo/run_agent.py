#!/usr/bin/env python3
"""
Demo: Full DeepCritical Agent Loop (Search + Judge + Orchestrator).

This script demonstrates Phase 4 functionality:
- Iterative Search (PubMed + Web)
- Evidence Evaluation (Judge Agent)
- Orchestration Loop
- Final Synthesis

Usage:
    # Run with Mock Judge (No API Key needed)
    uv run python examples/orchestrator_demo/run_agent.py "metformin cancer" --mock

    # Run with Real Judge (Requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
    uv run python examples/orchestrator_demo/run_agent.py "metformin cancer"
"""

import argparse
import asyncio
import os
import sys

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.orchestrator import Orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.tools.websearch import WebTool
from src.utils.models import OrchestratorConfig


async def main() -> None:
    """Run the agent demo."""
    parser = argparse.ArgumentParser(description="Run DeepCritical Agent CLI")
    parser.add_argument("query", help="Research query (e.g., 'metformin cancer')")
    parser.add_argument("--mock", action="store_true", help="Use Mock Judge (no API key needed)")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    args = parser.parse_args()

    # Check for keys if not mocking
    if not args.mock and not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or use --mock.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DeepCritical Agent Demo")
    print(f"Query: {args.query}")
    print(f"Mode: {'MOCK' if args.mock else 'REAL (LLM)'}")
    print(f"{ '='*60}\n")

    # 1. Setup Search Tools
    search_handler = SearchHandler(tools=[PubMedTool(), WebTool()], timeout=30.0)

    # 2. Setup Judge
    judge_handler: JudgeHandler | MockJudgeHandler
    if args.mock:
        judge_handler = MockJudgeHandler()
    else:
        judge_handler = JudgeHandler()

    # 3. Setup Orchestrator
    config = OrchestratorConfig(max_iterations=args.iterations)
    orchestrator = Orchestrator(
        search_handler=search_handler, judge_handler=judge_handler, config=config
    )

    # 4. Run Loop
    try:
        async for event in orchestrator.run(args.query):
            # Print event with icon
            print(event.to_markdown().replace("**", ""))

            # If we got data, print a snippet
            if event.type == "search_complete" and event.data:
                print(f"   -> Found {event.data.get('new_count', 0)} new items")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
