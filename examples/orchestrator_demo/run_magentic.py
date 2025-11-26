#!/usr/bin/env python3
"""
Demo: Magentic-One Orchestrator for DeepCritical.

This script demonstrates Phase 5 functionality:
- Multi-Agent Coordination (Searcher + Judge + Manager)
- Magentic-One Workflow

Usage:
    export OPENAI_API_KEY=...
    uv run python examples/orchestrator_demo/run_magentic.py "metformin cancer"
"""

import argparse
import asyncio
import os
import sys

from src.agent_factory.judges import JudgeHandler
from src.orchestrator_factory import create_orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.models import OrchestratorConfig


async def main() -> None:
    """Run the magentic agent demo."""
    parser = argparse.ArgumentParser(description="Run DeepCritical Magentic Agent")
    parser.add_argument("query", help="Research query (e.g., 'metformin cancer')")
    parser.add_argument("--iterations", type=int, default=10, help="Max rounds")
    args = parser.parse_args()

    # Check for keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: No API key found. Magentic requires real LLM.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DeepCritical Magentic Agent Demo")
    print(f"Query: {args.query}")
    print("Mode: MAGENTIC (Multi-Agent)")
    print(f"{ '='*60}\n")

    # 1. Setup Search Tools
    search_handler = SearchHandler(tools=[PubMedTool()], timeout=30.0)

    # 2. Setup Judge
    judge_handler = JudgeHandler()

    # 3. Setup Orchestrator via Factory
    config = OrchestratorConfig(max_iterations=args.iterations)
    orchestrator = create_orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
        mode="magentic",
    )

    if not orchestrator:
        print("Failed to create Magentic orchestrator. Is agent-framework installed?")
        sys.exit(1)

    # 4. Run Loop
    try:
        async for event in orchestrator.run(args.query):
            # Print event with icon
            # Clean up markdown for CLI
            msg_obj = event.message
            msg_text = ""
            if hasattr(msg_obj, "text"):
                msg_text = msg_obj.text
            else:
                msg_text = str(msg_obj)

            msg = msg_text.replace("\n", " ").replace("**", "")[:150]
            print(f"[{event.type.upper()}] {msg}...")

            if event.type == "complete":
                print("\n--- FINAL OUTPUT ---\n")
                print(msg_text)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
