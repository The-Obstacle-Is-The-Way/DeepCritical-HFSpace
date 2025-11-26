"""MCP tool wrappers for DeepCritical search tools.

These functions expose our search tools via MCP protocol.
Each function follows the MCP tool contract:
- Full type hints
- Google-style docstrings with Args section
- Formatted string returns
"""

from src.tools.biorxiv import BioRxivTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool

# Singleton instances (avoid recreating on each call)
_pubmed = PubMedTool()
_trials = ClinicalTrialsTool()
_biorxiv = BioRxivTool()


async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for peer-reviewed biomedical literature.

    Searches NCBI PubMed database for scientific papers matching your query.
    Returns titles, authors, abstracts, and citation information.

    Args:
        query: Search query (e.g., "metformin alzheimer", "drug repurposing cancer")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted search results with paper titles, authors, dates, and abstracts
    """
    max_results = max(1, min(50, max_results))  # Clamp to valid range

    results = await _pubmed.search(query, max_results)

    if not results:
        return f"No PubMed results found for: {query}"

    formatted = [f"## PubMed Results for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**Authors**: {', '.join(evidence.citation.authors[:3])}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_clinical_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for clinical trial data.

    Searches the ClinicalTrials.gov database for trials matching your query.
    Returns trial titles, phases, status, conditions, and interventions.

    Args:
        query: Search query (e.g., "metformin alzheimer", "diabetes phase 3")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted clinical trial information with NCT IDs, phases, and status
    """
    max_results = max(1, min(50, max_results))

    results = await _trials.search(query, max_results)

    if not results:
        return f"No clinical trials found for: {query}"

    formatted = [f"## Clinical Trials for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_biorxiv(query: str, max_results: int = 10) -> str:
    """Search bioRxiv/medRxiv for preprint research.

    Searches bioRxiv and medRxiv preprint servers for cutting-edge research.
    Note: Preprints are NOT peer-reviewed but contain the latest findings.

    Args:
        query: Search query (e.g., "metformin neuroprotection", "long covid treatment")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted preprint results with titles, authors, and abstracts
    """
    max_results = max(1, min(50, max_results))

    results = await _biorxiv.search(query, max_results)

    if not results:
        return f"No bioRxiv/medRxiv preprints found for: {query}"

    formatted = [f"## Preprint Results for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**Authors**: {', '.join(evidence.citation.authors[:3])}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_all_sources(query: str, max_per_source: int = 5) -> str:
    """Search all biomedical sources simultaneously.

    Performs parallel search across PubMed, ClinicalTrials.gov, and bioRxiv.
    This is the most comprehensive search option for drug repurposing research.

    Args:
        query: Search query (e.g., "metformin alzheimer", "aspirin cancer prevention")
        max_per_source: Maximum results per source (1-20, default 5)

    Returns:
        Combined results from all sources with source labels
    """
    import asyncio

    max_per_source = max(1, min(20, max_per_source))

    # Run all searches in parallel
    pubmed_task = search_pubmed(query, max_per_source)
    trials_task = search_clinical_trials(query, max_per_source)
    biorxiv_task = search_biorxiv(query, max_per_source)

    pubmed_results, trials_results, biorxiv_results = await asyncio.gather(
        pubmed_task, trials_task, biorxiv_task, return_exceptions=True
    )

    formatted = [f"# Comprehensive Search: {query}\n"]

    # Add each result section (handle exceptions gracefully)
    if isinstance(pubmed_results, str):
        formatted.append(pubmed_results)
    else:
        formatted.append(f"## PubMed\n*Error: {pubmed_results}*\n")

    if isinstance(trials_results, str):
        formatted.append(trials_results)
    else:
        formatted.append(f"## Clinical Trials\n*Error: {trials_results}*\n")

    if isinstance(biorxiv_results, str):
        formatted.append(biorxiv_results)
    else:
        formatted.append(f"## Preprints\n*Error: {biorxiv_results}*\n")

    return "\n---\n".join(formatted)


async def analyze_hypothesis(
    drug: str,
    condition: str,
    evidence_summary: str,
) -> str:
    """Perform statistical analysis of drug repurposing hypothesis using Modal.

    Executes AI-generated Python code in a secure Modal sandbox to analyze
    the statistical evidence for a drug repurposing hypothesis.

    Args:
        drug: The drug being evaluated (e.g., "metformin")
        condition: The target condition (e.g., "Alzheimer's disease")
        evidence_summary: Summary of evidence to analyze

    Returns:
        Analysis result with verdict (SUPPORTED/REFUTED/INCONCLUSIVE) and statistics
    """
    from src.services.statistical_analyzer import get_statistical_analyzer
    from src.utils.config import settings
    from src.utils.models import Citation, Evidence

    if not settings.modal_available:
        return "Error: Modal credentials not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."

    # Create evidence from summary
    evidence = [
        Evidence(
            content=evidence_summary,
            citation=Citation(
                source="pubmed",
                title=f"Evidence for {drug} in {condition}",
                url="https://example.com",
                date="2024-01-01",
                authors=["User Provided"],
            ),
            relevance=0.9,
        )
    ]

    analyzer = get_statistical_analyzer()
    result = await analyzer.analyze(
        query=f"Can {drug} treat {condition}?",
        evidence=evidence,
        hypothesis={"drug": drug, "target": "unknown", "pathway": "unknown", "effect": condition},
    )

    return f"""## Statistical Analysis: {drug} for {condition}

### Verdict: **{result.verdict}**
**Confidence**: {result.confidence:.0%}

### Key Findings
{chr(10).join(f"- {f}" for f in result.key_findings) or "- No specific findings extracted"}

### Execution Output
```
{result.execution_output}
```

### Generated Code
```python
{result.code_generated}
```

**Executed in Modal Sandbox** - Isolated, secure, reproducible.
"""
