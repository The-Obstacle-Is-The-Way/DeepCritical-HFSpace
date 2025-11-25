#!/usr/bin/env python3
"""
Demo: Hypothesis Generation (Phase 7).

This script demonstrates mechanistic hypothesis generation:
- Drug -> Target -> Pathway -> Effect reasoning
- Knowledge gap identification
- Search query suggestions for targeted research

Usage:
    # Requires OPENAI_API_KEY or ANTHROPIC_API_KEY
    uv run python examples/hypothesis_demo/run_hypothesis.py

    # With custom drug query
    uv run python examples/hypothesis_demo/run_hypothesis.py "aspirin heart disease"
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from src.agents.hypothesis_agent import HypothesisAgent
from src.utils.models import Citation, Evidence


def create_metformin_evidence() -> list[Evidence]:
    """Create sample evidence about metformin for hypothesis generation."""
    return [
        Evidence(
            content=(
                "Metformin activates AMP-activated protein kinase (AMPK), a master regulator "
                "of cellular energy homeostasis. AMPK activation leads to inhibition of mTOR "
                "signaling, reducing protein synthesis and cell proliferation."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin and AMPK: mechanisms of action",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
                authors=["Zhang L", "Wang H"],
            ),
        ),
        Evidence(
            content=(
                "In Alzheimer's disease models, AMPK activation by metformin reduced tau "
                "phosphorylation and amyloid-beta accumulation. These effects correlated "
                "with improved cognitive function in transgenic mice."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin neuroprotective effects in AD models",
                url="https://pubmed.ncbi.nlm.nih.gov/23456/",
                date="2024",
                authors=["Kim J", "Lee S", "Park M"],
            ),
        ),
        Evidence(
            content=(
                "Clinical observational studies show diabetic patients on metformin have "
                "30-40% reduced incidence of Alzheimer's disease compared to those on "
                "other diabetes medications."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin use and dementia risk: population study",
                url="https://pubmed.ncbi.nlm.nih.gov/34567/",
                date="2023",
                authors=["Smith A", "Johnson B"],
            ),
        ),
        Evidence(
            content=(
                "mTOR inhibition has emerged as a key therapeutic target in neurodegenerative "
                "diseases. Rapamycin and metformin both reduce mTOR activity, though through "
                "different upstream mechanisms."
            ),
            citation=Citation(
                source="pubmed",
                title="mTOR pathway in neurodegeneration",
                url="https://pubmed.ncbi.nlm.nih.gov/45678/",
                date="2022",
                authors=["Brown C", "Davis D"],
            ),
        ),
        Evidence(
            content=(
                "Metformin crosses the blood-brain barrier and accumulates in the hippocampus "
                "and cortex. Brain concentrations sufficient for AMPK activation are achieved "
                "at standard diabetic doses."
            ),
            citation=Citation(
                source="pubmed",
                title="Pharmacokinetics of metformin in brain tissue",
                url="https://pubmed.ncbi.nlm.nih.gov/56789/",
                date="2023",
                authors=["Wilson E"],
            ),
        ),
    ]


def create_sildenafil_evidence() -> list[Evidence]:
    """Create sample evidence about sildenafil (Viagra) for hypothesis generation."""
    return [
        Evidence(
            content=(
                "Sildenafil inhibits phosphodiesterase type 5 (PDE5), preventing breakdown "
                "of cGMP. Elevated cGMP causes smooth muscle relaxation and vasodilation "
                "in pulmonary vasculature."
            ),
            citation=Citation(
                source="pubmed",
                title="PDE5 inhibition mechanism of sildenafil",
                url="https://pubmed.ncbi.nlm.nih.gov/67890/",
                date="2022",
                authors=["Miller F"],
            ),
        ),
        Evidence(
            content=(
                "In pulmonary arterial hypertension (PAH), sildenafil reduces pulmonary "
                "vascular resistance and improves exercise capacity. FDA approved for PAH "
                "under brand name Revatio."
            ),
            citation=Citation(
                source="pubmed",
                title="Sildenafil in pulmonary hypertension treatment",
                url="https://pubmed.ncbi.nlm.nih.gov/78901/",
                date="2023",
                authors=["Garcia R", "Martinez L"],
            ),
        ),
        Evidence(
            content=(
                "PDE5 is expressed in cardiac myocytes. Sildenafil has shown cardioprotective "
                "effects in animal models of heart failure by enhancing nitric oxide-cGMP "
                "signaling in the myocardium."
            ),
            citation=Citation(
                source="pubmed",
                title="Cardiac effects of PDE5 inhibition",
                url="https://pubmed.ncbi.nlm.nih.gov/89012/",
                date="2024",
                authors=["Thompson K"],
            ),
        ),
    ]


async def run_hypothesis_demo(query: str) -> None:
    """Run the hypothesis generation demo."""
    print(f"\n{'='*60}")
    print("DeepCritical Hypothesis Agent Demo (Phase 7)")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Select appropriate evidence based on query
    if "sildenafil" in query.lower() or "viagra" in query.lower():
        evidence = create_sildenafil_evidence()
        print("Using: Sildenafil evidence set (3 papers)")
    else:
        evidence = create_metformin_evidence()
        print("Using: Metformin evidence set (5 papers)")

    # Create evidence store (shared context between agents)
    evidence_store: dict[str, Any] = {"current": evidence, "hypotheses": []}

    # Create hypothesis agent
    agent = HypothesisAgent(evidence_store)

    print("\nGenerating mechanistic hypotheses...\n")
    print("-" * 60)

    # Run hypothesis generation
    response = await agent.run(query)

    # Print the formatted response
    print(response.messages[0].text)

    print("-" * 60)

    # Show stored hypotheses
    hypotheses = evidence_store.get("hypotheses", [])
    print(f"\n{len(hypotheses)} hypotheses stored in evidence_store")

    if hypotheses:
        print("\nHypothesis search queries generated:")
        for h in hypotheses:
            queries = h.to_search_queries()
            print(f"  - {h.drug} -> {h.target}: {queries[:2]}")


async def main() -> None:
    """Run the demo."""
    parser = argparse.ArgumentParser(description="Hypothesis Generation Demo")
    parser.add_argument(
        "query",
        nargs="?",
        default="metformin Alzheimer's disease",
        help="Research query (default: 'metformin Alzheimer\\'s disease')",
    )
    args = parser.parse_args()

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: Hypothesis generation requires an LLM.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)

    await run_hypothesis_demo(args.query)

    print("\n" + "=" * 60)
    print("Demo complete! The Hypothesis Agent:")
    print("  - Analyzes evidence to find Drug -> Target -> Pathway -> Effect chains")
    print("  - Identifies knowledge gaps in current evidence")
    print("  - Suggests targeted search queries to test hypotheses")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
