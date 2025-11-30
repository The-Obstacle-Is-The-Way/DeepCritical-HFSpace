"""Tool functions for Magentic agents.

These functions are decorated with @ai_function to be callable by the ChatAgent's internal LLM.
They also interact with the thread-safe MagenticState to persist evidence.
"""

from agent_framework import ai_function

from src.agents.state import get_magentic_state
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool

# Singleton tool instances (stateless wrappers)
_pubmed = PubMedTool()
_clinicaltrials = ClinicalTrialsTool()
_europepmc = EuropePMCTool()


@ai_function  # type: ignore[arg-type, misc, untyped-decorator]
async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical research papers.

    Use this tool to find peer-reviewed scientific literature about
    drugs, diseases, mechanisms of action, and clinical studies.

    Args:
        query: Search keywords (e.g., "testosterone libido mechanism")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of papers with titles, abstracts, and citations
    """
    state = get_magentic_state()

    # 1. Execute raw search
    results = await _pubmed.search(query, max_results)
    if not results:
        return f"No PubMed results found for: {query}"

    # 2. Store in Memory (handles dedup and persistence)
    # ResearchMemory handles semantic deduplication and persistence
    new_ids = await state.memory.store_evidence(results)
    new_count = len(new_ids)

    # 3. Context Expansion (The "Digital Twin" Brain)
    # Combine what we just found with what we already know is relevant
    display_results = list(results)

    # Search for related context in the memory (previous searches)
    related = await state.memory.get_relevant_evidence(n=3)

    # Add related items if they aren't already in the results
    current_urls = {r.citation.url for r in display_results}
    for item in related:
        if item.citation.url not in current_urls:
            display_results.append(item)

    # 4. Format Output for LLM
    output = [f"Found {len(results)} results ({new_count} new stored):\n"]

    # Limit display to avoid context window overflow, but state has everything
    limit = min(len(display_results), max_results + 3)

    for i, r in enumerate(display_results[:limit], 1):
        title = r.citation.title
        date = r.citation.date
        source = r.citation.source
        content_clean = r.content[:300].replace("\n", " ")
        url = r.citation.url

        output.append(f"{i}. **{title}** ({date})")
        output.append(f"   Source: {source} | {url}")
        output.append(f"   {content_clean}...")
        output.append("")

    return "\n".join(output)


@ai_function  # type: ignore[arg-type, misc, untyped-decorator]
async def search_clinical_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for clinical studies.

    Use this tool to find ongoing and completed clinical trials
    for potential interventions.

    Args:
        query: Search terms (e.g., "sildenafil phase 3")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of clinical trials with status and details
    """
    state = get_magentic_state()

    results = await _clinicaltrials.search(query, max_results)
    if not results:
        return f"No clinical trials found for: {query}"

    # Update state
    new_ids = await state.memory.store_evidence(results)
    new_count = len(new_ids)

    output = [f"Found {len(results)} clinical trials ({new_count} new stored):\n"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.citation.title
        date = r.citation.date
        source = r.citation.source
        content_clean = r.content[:300].replace("\n", " ")
        url = r.citation.url

        output.append(f"{i}. **{title}**")
        output.append(f"   Status: {source} | Date: {date}")
        output.append(f"   {content_clean}...")
        output.append(f"   URL: {url}\n")

    return "\n".join(output)


@ai_function  # type: ignore[arg-type, misc, untyped-decorator]
async def search_preprints(query: str, max_results: int = 10) -> str:
    """Search Europe PMC for preprints and papers.

    Use this tool to find the latest research including preprints
    from bioRxiv, medRxiv, and peer-reviewed papers.

    Args:
        query: Search terms (e.g., "flibanserin HSDD preprint")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of papers with abstracts and links
    """
    state = get_magentic_state()

    results = await _europepmc.search(query, max_results)
    if not results:
        return f"No papers found for: {query}"

    # Update state
    new_ids = await state.memory.store_evidence(results)
    new_count = len(new_ids)

    output = [f"Found {len(results)} papers ({new_count} new stored):\n"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.citation.title
        date = r.citation.date
        source = r.citation.source
        content_clean = r.content[:300].replace("\n", " ")
        url = r.citation.url

        output.append(f"{i}. **{title}**")
        output.append(f"   Source: {source} | Date: {date}")
        output.append(f"   {content_clean}...")
        output.append(f"   URL: {url}\n")

    return "\n".join(output)


@ai_function  # type: ignore[arg-type, misc, untyped-decorator]
async def get_bibliography() -> str:
    """Get the full list of collected evidence for the bibliography.

    Use this tool when generating the final report to get the complete
    list of references.

    Returns:
        Formatted bibliography string.
    """
    state = get_magentic_state()
    all_evidence = state.memory.get_all_evidence()

    if not all_evidence:
        return "No evidence collected."

    output = ["## References"]
    for i, ev in enumerate(all_evidence, 1):
        output.append(f"{i}. {ev.citation.formatted}")
        output.append(f"   URL: {ev.citation.url}")

    return "\n".join(output)
