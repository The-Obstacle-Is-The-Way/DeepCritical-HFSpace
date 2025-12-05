"""MCP tool wrappers for DeepBoner search tools.

These functions expose our search tools via MCP protocol.
Each function follows the MCP tool contract:
- Full type hints
- Google-style docstrings with Args section
- Formatted string returns
"""

from src.config.domain import get_domain_config
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool

# Singleton instances (avoid recreating on each call)
_pubmed = PubMedTool()
_trials = ClinicalTrialsTool()
_europepmc = EuropePMCTool()


async def search_pubmed(query: str, max_results: int = 10, domain: str = "sexual_health") -> str:
    """Search PubMed for peer-reviewed biomedical literature.

    Searches NCBI PubMed database for scientific papers matching your query.
    Returns titles, authors, abstracts, and citation information.

    Args:
        query: Search query (e.g., "testosterone libido")
        max_results: Maximum results to return (1-50, default 10)
        domain: Research domain (defaults to "sexual_health")

    Returns:
        Formatted search results with paper titles, authors, dates, and abstracts
    """
    max_results = max(1, min(50, max_results))  # Clamp to valid range
    config = get_domain_config(domain)

    results = await _pubmed.search(query, max_results)

    if not results:
        return f"No PubMed results found for: {query}"

    formatted = [f"## PubMed Results for: {query} ({config.name})\n"]
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
        query: Search query (e.g., "testosterone hypoactive desire", "sildenafil phase 3")
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


async def search_europepmc(query: str, max_results: int = 10) -> str:
    """Search Europe PMC for preprints and papers.

    Searches Europe PMC, which includes bioRxiv, medRxiv, and peer-reviewed content.
    Useful for finding cutting-edge preprints and open access papers.

    Args:
        query: Search query (e.g., "flibanserin mechanism", "erectile dysfunction novel treatment")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted results with titles, authors, and abstracts
    """
    max_results = max(1, min(50, max_results))

    results = await _europepmc.search(query, max_results)

    if not results:
        return f"No Europe PMC results found for: {query}"

    formatted = [f"## Europe PMC Results for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**Authors**: {', '.join(evidence.citation.authors[:3])}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_all_sources(
    query: str, max_per_source: int = 5, domain: str = "sexual_health"
) -> str:
    """Search all biomedical sources simultaneously.

    Performs parallel search across PubMed, ClinicalTrials.gov, and Europe PMC.
    This is the most comprehensive search option for biomedical research.

    Args:
        query: Search query (e.g., "testosterone replacement therapy", "HSDD treatment")
        max_per_source: Maximum results per source (1-20, default 5)
        domain: Research domain (defaults to "sexual_health")

    Returns:
        Combined results from all sources with source labels
    """
    import asyncio

    max_per_source = max(1, min(20, max_per_source))
    config = get_domain_config(domain)

    # Run all searches in parallel
    pubmed_task = search_pubmed(query, max_per_source, domain)
    trials_task = search_clinical_trials(query, max_per_source)
    europepmc_task = search_europepmc(query, max_per_source)

    pubmed_results, trials_results, europepmc_results = await asyncio.gather(
        pubmed_task, trials_task, europepmc_task, return_exceptions=True
    )

    formatted = [f"# Comprehensive Search: {query} ({config.name})\n"]

    # Add each result section (handle exceptions gracefully)
    if isinstance(pubmed_results, str):
        formatted.append(pubmed_results)
    else:
        formatted.append(f"## PubMed\n*Error: {pubmed_results}*\n")

    if isinstance(trials_results, str):
        formatted.append(trials_results)
    else:
        formatted.append(f"## Clinical Trials\n*Error: {trials_results}*\n")

    if isinstance(europepmc_results, str):
        formatted.append(europepmc_results)
    else:
        formatted.append(f"## Europe PMC\n*Error: {europepmc_results}*\n")

    return "\n---\n".join(formatted)
