#!/usr/bin/env python3
"""
Demo: Full Stack DeepCritical Agent (Phases 1-8).

This script demonstrates the COMPLETE drug repurposing research pipeline:
- Phase 2: Search (PubMed + Web)
- Phase 6: Embeddings (Semantic deduplication)
- Phase 7: Hypothesis (Mechanistic reasoning)
- Phase 3: Judge (Evidence assessment)
- Phase 8: Report (Structured scientific report)

Usage:
    # Full demo with real searches and LLM (requires API keys)
    uv run python examples/full_stack_demo/run_full.py "metformin Alzheimer's"

    # Mock mode - demonstrates pipeline without API calls
    uv run python examples/full_stack_demo/run_full.py --mock

    # With specific iterations
    uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" --iterations 2
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from src.utils.models import Citation, Evidence, MechanismHypothesis


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step: int, name: str) -> None:
    """Print a step indicator."""
    print(f"\n[Step {step}] {name}")
    print("-" * 50)


def create_mock_evidence() -> list[Evidence]:
    """Create comprehensive mock evidence for demo without API calls."""
    return [
        Evidence(
            content=(
                "Metformin, a first-line treatment for type 2 diabetes, activates "
                "AMP-activated protein kinase (AMPK). AMPK is a master metabolic "
                "regulator that inhibits mTOR signaling, reducing protein synthesis "
                "and cell proliferation. This mechanism has implications beyond "
                "glucose control."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin activates AMPK through LKB1-dependent mechanisms",
                url="https://pubmed.ncbi.nlm.nih.gov/19001324/",
                date="2023-06",
                authors=["Zhang L", "Wang H", "Chen Y"],
            ),
        ),
        Evidence(
            content=(
                "In transgenic mouse models of Alzheimer's disease, metformin treatment "
                "reduced tau phosphorylation by 45% and decreased amyloid-beta plaque "
                "formation. Treated mice showed improved performance on Morris water "
                "maze tests, suggesting preserved spatial memory."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin ameliorates tau pathology in AD mouse models",
                url="https://pubmed.ncbi.nlm.nih.gov/31256789/",
                date="2024-01",
                authors=["Kim J", "Lee S", "Park M", "Tanaka K"],
            ),
        ),
        Evidence(
            content=(
                "A population-based cohort study of 100,000 diabetic patients found "
                "that metformin users had 35% lower risk of developing Alzheimer's "
                "disease compared to sulfonylurea users (HR=0.65, 95% CI: 0.58-0.73). "
                "The protective effect increased with duration of use."
            ),
            citation=Citation(
                source="pubmed",
                title="Metformin and dementia risk: UK Biobank analysis",
                url="https://pubmed.ncbi.nlm.nih.gov/34567890/",
                date="2023-09",
                authors=["Smith A", "Johnson B", "Williams C"],
            ),
        ),
        Evidence(
            content=(
                "mTOR hyperactivation is observed in Alzheimer's disease brain tissue. "
                "mTOR inhibition by rapamycin or metformin promotes autophagy, which "
                "clears misfolded proteins including tau and amyloid-beta aggregates. "
                "This suggests a common therapeutic pathway."
            ),
            citation=Citation(
                source="pubmed",
                title="mTOR-autophagy axis in neurodegeneration",
                url="https://pubmed.ncbi.nlm.nih.gov/32109876/",
                date="2023-03",
                authors=["Brown C", "Davis D", "Miller E"],
            ),
        ),
        Evidence(
            content=(
                "Metformin crosses the blood-brain barrier via organic cation "
                "transporters (OCT1, OCT2). CSF concentrations reach approximately "
                "1-2% of plasma levels, sufficient for AMPK activation in neurons. "
                "Brain accumulation is observed in hippocampus and prefrontal cortex."
            ),
            citation=Citation(
                source="pubmed",
                title="Brain pharmacokinetics of metformin in humans",
                url="https://pubmed.ncbi.nlm.nih.gov/35678901/",
                date="2024-02",
                authors=["Wilson E", "Garcia F"],
            ),
        ),
        Evidence(
            content=(
                "Phase 2 clinical trial (NCT04098666) showed metformin 2000mg/day "
                "for 12 months slowed cognitive decline by 18% compared to placebo "
                "in patients with mild cognitive impairment. Biomarker analysis "
                "showed reduced CSF tau levels in the treatment group."
            ),
            citation=Citation(
                source="web",
                title="Metformin for Alzheimer's prevention trial results",
                url="https://clinicaltrials.gov/ct2/show/NCT04098666",
                date="2024-03",
                authors=["NIH Clinical Center"],
            ),
        ),
    ]


def create_mock_hypotheses() -> list[MechanismHypothesis]:
    """Create mock hypotheses for demonstration."""
    return [
        MechanismHypothesis(
            drug="Metformin",
            target="AMPK",
            pathway="mTOR inhibition -> Autophagy activation",
            effect="Clearance of tau and amyloid-beta aggregates",
            confidence=0.85,
            supporting_evidence=[
                "https://pubmed.ncbi.nlm.nih.gov/19001324/",
                "https://pubmed.ncbi.nlm.nih.gov/32109876/",
            ],
            contradicting_evidence=[],
            search_suggestions=[
                "AMPK autophagy neurodegeneration",
                "metformin tau clearance",
            ],
        ),
        MechanismHypothesis(
            drug="Metformin",
            target="Glucose metabolism",
            pathway="Improved neuronal energy homeostasis",
            effect="Reduced oxidative stress and neuroinflammation",
            confidence=0.70,
            supporting_evidence=["https://pubmed.ncbi.nlm.nih.gov/31256789/"],
            contradicting_evidence=[],
            search_suggestions=[
                "metformin brain glucose metabolism",
                "neuronal insulin resistance alzheimer",
            ],
        ),
    ]


async def run_mock_demo() -> None:
    """Run full pipeline with mock data (no API keys needed)."""
    print_header("DeepCritical Full Stack Demo (MOCK MODE)")
    print("Running with synthetic data - no API keys required.\n")

    evidence = create_mock_evidence()
    hypotheses = create_mock_hypotheses()

    # Step 1: Show evidence
    print_step(1, "SEARCH (Phase 2) - Evidence Collection")
    print(f"Collected {len(evidence)} pieces of evidence:\n")
    for i, e in enumerate(evidence, 1):
        print(f"  [{i}] {e.citation.source.upper()}: {e.citation.title[:50]}...")
        print(f"      {e.content[:80]}...")
        print()

    # Step 2: Embedding deduplication
    print_step(2, "EMBEDDINGS (Phase 6) - Semantic Deduplication")
    try:
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()
        unique = await service.deduplicate(evidence, threshold=0.85)
        print(f"Original: {len(evidence)} papers")
        print(f"After deduplication: {len(unique)} unique papers")
        print("(Semantic duplicates removed by meaning, not just URL)")
    except ImportError:
        print("Embedding dependencies not installed - skipping deduplication")
        unique = evidence

    # Step 3: Hypothesis generation
    print_step(3, "HYPOTHESIS (Phase 7) - Mechanistic Reasoning")
    print(f"Generated {len(hypotheses)} hypotheses:\n")
    for i, h in enumerate(hypotheses, 1):
        print(f"  Hypothesis {i} (Confidence: {h.confidence:.0%})")
        print(f"  {h.drug} -> {h.target} -> {h.pathway} -> {h.effect}")
        print(f"  Suggested searches: {', '.join(h.search_suggestions)}")
        print()

    # Step 4: Judge assessment
    print_step(4, "JUDGE (Phase 3) - Evidence Assessment")
    print("Assessment Results:")
    print("  Mechanism Score:  8/10 (Strong mechanistic evidence)")
    print("  Clinical Score:   7/10 (Phase 2 trial + observational data)")
    print("  Confidence:       75%")
    print("  Recommendation:   SYNTHESIZE (Evidence sufficient)")
    print()

    # Step 5: Report generation
    print_step(5, "REPORT (Phase 8) - Structured Scientific Report")

    report = f"""
# Drug Repurposing Analysis: Metformin for Alzheimer's Disease

## Executive Summary
This analysis evaluated metformin as a potential therapeutic for Alzheimer's
disease. Evidence from {len(unique)} sources supports a plausible mechanism
through AMPK activation and mTOR inhibition, leading to enhanced autophagy
and clearance of pathological protein aggregates. Clinical data shows
promising risk reduction in observational studies and early trial results.

## Research Question
Can metformin, a type 2 diabetes medication, be repurposed for the prevention
or treatment of Alzheimer's disease?

## Methodology
- Searched PubMed and web sources for "metformin Alzheimer's disease"
- Applied semantic deduplication to remove redundant findings
- Generated mechanistic hypotheses using LLM reasoning
- Evaluated evidence quality with structured assessment

## Hypotheses Tested
- **Metformin -> AMPK -> mTOR inhibition -> Neuroprotection** (SUPPORTED)
  - 4 supporting papers, 0 contradicting
- **Metformin -> Glucose metabolism -> Reduced oxidative stress** (PARTIAL)
  - 2 supporting papers, requires more investigation

## Mechanistic Findings
Strong evidence supports AMPK activation as the primary mechanism. Metformin
crosses the blood-brain barrier and achieves therapeutic concentrations in
hippocampus and cortex. Downstream effects include:
- mTOR inhibition
- Autophagy activation
- Tau dephosphorylation
- Amyloid-beta clearance

## Clinical Findings
- Observational: 35% risk reduction (HR=0.65, n=100,000)
- Preclinical: 45% reduction in tau phosphorylation in AD mice
- Phase 2 trial: 18% slower cognitive decline vs placebo

## Drug Candidates
- **Metformin** - Primary candidate with established safety profile

## Limitations
- Abstract-level analysis only
- Observational data subject to confounding
- Limited RCT data available
- Optimal dosing for neuroprotection unclear

## Conclusion
Metformin shows strong potential for Alzheimer's disease prevention/treatment.
The AMPK-mTOR-autophagy mechanism is well-supported. Recommend Phase 3 trials
with cognitive endpoints.

## References
"""
    max_authors_display = 2
    for i, e in enumerate(unique[:6], 1):
        authors = ", ".join(e.citation.authors[:max_authors_display])
        if len(e.citation.authors) > max_authors_display:
            authors += " et al."
        ref_line = (
            f"{i}. {authors}. *{e.citation.title}*. "
            f"{e.citation.source.upper()} ({e.citation.date}). "
            f"[Link]({e.citation.url})"
        )
        report += ref_line + "\n"

    report += f"""
---
*Report generated from {len(unique)} papers across 3 search iterations.
Confidence: 75%*
"""

    print(report)


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


async def run_real_demo(query: str, max_iterations: int) -> None:
    """Run full pipeline with real API calls."""
    print_header("DeepCritical Full Stack Demo")
    print(f"Query: {query}")
    print(f"Max iterations: {max_iterations}")
    print("Mode: REAL (Live API calls)\n")

    # Import real components
    from src.agent_factory.judges import JudgeHandler
    from src.agents.hypothesis_agent import HypothesisAgent
    from src.agents.report_agent import ReportAgent
    from src.services.embeddings import EmbeddingService
    from src.tools.pubmed import PubMedTool
    from src.tools.search_handler import SearchHandler
    from src.tools.websearch import WebTool

    # Initialize services
    embedding_service = EmbeddingService()
    search_handler = SearchHandler(tools=[PubMedTool(), WebTool()], timeout=30.0)
    judge_handler = JudgeHandler()

    # Shared evidence store
    evidence_store: dict[str, Any] = {"current": [], "hypotheses": [], "iteration_count": 0}
    all_evidence: list[Evidence] = []

    for iteration in range(1, max_iterations + 1):
        print_step(iteration, f"ITERATION {iteration}/{max_iterations}")

        # Step 1: Search
        print("\n[Search] Querying PubMed and Web...")
        all_evidence = await _run_search_iteration(
            query, iteration, evidence_store, all_evidence, search_handler, embedding_service
        )

        # Step 2: Generate hypotheses (first iteration only)
        if iteration == 1:
            print("\n[Hypothesis] Generating mechanistic hypotheses...")
            hypothesis_agent = HypothesisAgent(evidence_store, embedding_service)
            hyp_response = await hypothesis_agent.run(query)
            print(hyp_response.messages[0].text[:500] + "...")

        # Step 3: Judge
        print("\n[Judge] Assessing evidence quality...")
        assessment = await judge_handler.assess(query, all_evidence)
        print(f"  Mechanism: {assessment.details.mechanism_score}/10")
        print(f"  Clinical: {assessment.details.clinical_evidence_score}/10")
        print(f"  Recommendation: {assessment.recommendation}")

        if assessment.recommendation == "synthesize":
            print("\n[Judge says] Evidence sufficient! Generating report...")
            evidence_store["last_assessment"] = assessment.details.model_dump()
            break

        next_queries = assessment.next_search_queries[:2]
        print(f"\n[Judge says] Need more evidence. Next queries: {next_queries}")
        query = assessment.next_search_queries[0] if assessment.next_search_queries else query

    # Step 4: Generate report
    print_step(iteration + 1, "REPORT GENERATION")
    report_agent = ReportAgent(evidence_store, embedding_service)
    report_response = await report_agent.run(query)

    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    print(report_response.messages[0].text)


async def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="DeepCritical Full Stack Demo (Phases 1-8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Mock mode (no API keys)
    uv run python examples/full_stack_demo/run_full.py --mock

    # Real mode with metformin query
    uv run python examples/full_stack_demo/run_full.py "metformin alzheimer"

    # Sildenafil for heart failure
    uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" -i 3
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="metformin Alzheimer's disease",
        help="Research query",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock data (no API keys needed)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=2,
        help="Max search iterations (default: 2)",
    )

    args = parser.parse_args()

    if args.mock:
        await run_mock_demo()
    else:
        # Check for API keys
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            print("Error: Real mode requires OPENAI_API_KEY or ANTHROPIC_API_KEY")
            print("Use --mock for demo without API keys.")
            sys.exit(1)

        await run_real_demo(args.query, args.iterations)

    print("\n" + "=" * 70)
    print("  DeepCritical Full Stack Demo Complete!")
    print("  Phases demonstrated: Foundation -> Search -> Judge -> UI ->")
    print("                       Magentic -> Embeddings -> Hypothesis -> Report")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
