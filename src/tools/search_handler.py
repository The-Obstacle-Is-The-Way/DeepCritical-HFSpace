"""Search handler - orchestrates multiple search tools."""

import asyncio
import re
from typing import TYPE_CHECKING, cast

import structlog

from src.tools.base import SearchTool
from src.utils.exceptions import SearchError
from src.utils.models import Evidence, SearchResult, SourceName

if TYPE_CHECKING:
    from src.utils.models import Evidence

logger = structlog.get_logger()


def extract_paper_id(evidence: "Evidence") -> str | None:
    """Extract unique paper identifier from Evidence.

    Strategy:
    1. Check metadata.pmid first (OpenAlex provides this)
    2. Fall back to URL pattern matching

    Supports:
    - PubMed: https://pubmed.ncbi.nlm.nih.gov/12345678/
    - Europe PMC MED: https://europepmc.org/article/MED/12345678
    - Europe PMC PMC: https://europepmc.org/article/PMC/PMC1234567
    - Europe PMC PPR: https://europepmc.org/article/PPR/PPR123456
    - Europe PMC PAT: https://europepmc.org/article/PAT/WO8601415
    - DOI: https://doi.org/10.1234/...
    - OpenAlex: https://openalex.org/W1234567890
    - ClinicalTrials: https://clinicaltrials.gov/study/NCT12345678
    - ClinicalTrials (legacy): https://clinicaltrials.gov/ct2/show/NCT12345678
    """
    url = evidence.citation.url
    metadata = evidence.metadata or {}

    # Strategy 1: Check metadata.pmid (from OpenAlex)
    if pmid := metadata.get("pmid"):
        return f"PMID:{pmid}"

    # Strategy 2: URL pattern matching

    # PubMed URL pattern
    pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if pmid_match:
        return f"PMID:{pmid_match.group(1)}"

    # Europe PMC MED pattern (same as PMID)
    epmc_med_match = re.search(r"europepmc\.org/article/MED/(\d+)", url)
    if epmc_med_match:
        return f"PMID:{epmc_med_match.group(1)}"

    # Europe PMC PMC pattern (PubMed Central ID - different from PMID!)
    epmc_pmc_match = re.search(r"europepmc\.org/article/PMC/(PMC\d+)", url)
    if epmc_pmc_match:
        return f"PMCID:{epmc_pmc_match.group(1)}"

    # Europe PMC PPR pattern (Preprint ID - unique per preprint)
    epmc_ppr_match = re.search(r"europepmc\.org/article/PPR/(PPR\d+)", url)
    if epmc_ppr_match:
        return f"PPRID:{epmc_ppr_match.group(1)}"

    # Europe PMC PAT pattern (Patent ID - e.g., WO8601415, EP1234567)
    epmc_pat_match = re.search(r"europepmc\.org/article/PAT/([A-Z]{2}\d+)", url)
    if epmc_pat_match:
        return f"PATID:{epmc_pat_match.group(1)}"

    # DOI pattern (normalize trailing slash/characters)
    doi_match = re.search(r"doi\.org/(10\.\d+/[^\s\]>]+)", url)
    if doi_match:
        doi = doi_match.group(1).rstrip("/")
        return f"DOI:{doi}"

    # OpenAlex ID pattern (fallback if no PMID in metadata)
    openalex_match = re.search(r"openalex\.org/(W\d+)", url)
    if openalex_match:
        return f"OAID:{openalex_match.group(1)}"

    # ClinicalTrials NCT ID (modern format)
    nct_match = re.search(r"clinicaltrials\.gov/study/(NCT\d+)", url)
    if nct_match:
        return f"NCT:{nct_match.group(1)}"

    # ClinicalTrials NCT ID (legacy format)
    nct_legacy_match = re.search(r"clinicaltrials\.gov/ct2/show/(NCT\d+)", url)
    if nct_legacy_match:
        return f"NCT:{nct_legacy_match.group(1)}"

    return None


def deduplicate_evidence(evidence_list: list["Evidence"]) -> list["Evidence"]:
    """Remove duplicate evidence based on paper ID.

    Deduplication priority:
    1. PubMed (authoritative source)
    2. Europe PMC (full text links)
    3. OpenAlex (citation data)
    4. ClinicalTrials (unique, never duplicated)

    Returns:
        Deduplicated list preserving source priority order.
    """
    seen_ids: set[str] = set()
    unique: list[Evidence] = []

    # Sort by source priority (PubMed first)
    source_priority = {"pubmed": 0, "europepmc": 1, "openalex": 2, "clinicaltrials": 3}
    sorted_evidence = sorted(
        evidence_list, key=lambda e: source_priority.get(e.citation.source, 99)
    )

    for evidence in sorted_evidence:
        paper_id = extract_paper_id(evidence)

        if paper_id is None:
            # Can't identify - keep it (conservative)
            unique.append(evidence)
            continue

        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique.append(evidence)

    return unique


class SearchHandler:
    """Orchestrates parallel searches across multiple tools."""

    def __init__(self, tools: list[SearchTool], timeout: float = 30.0) -> None:
        """
        Initialize the search handler.

        Args:
            tools: List of search tools to use
            timeout: Timeout for each search in seconds
        """
        self.tools = tools
        self.timeout = timeout

    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        """
        Execute search across all tools in parallel.

        Args:
            query: The search query
            max_results_per_tool: Max results from each tool

        Returns:
            SearchResult containing all evidence and metadata
        """
        logger.info("Starting search", query=query, tools=[t.name for t in self.tools])

        # Create tasks for parallel execution
        tasks = [
            self._search_with_timeout(tool, query, max_results_per_tool) for tool in self.tools
        ]

        # Gather results (don't fail if one tool fails)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_evidence: list[Evidence] = []
        sources_searched: list[SourceName] = []
        errors: list[str] = []

        for tool, result in zip(self.tools, results, strict=True):
            if isinstance(result, Exception):
                errors.append(f"{tool.name}: {result!s}")
                logger.warning("Search tool failed", tool=tool.name, error=str(result))
            else:
                # Cast result to list[Evidence] as we know it succeeded
                success_result = cast(list[Evidence], result)
                all_evidence.extend(success_result)

                # Cast tool.name to SourceName (centralized type from models)
                tool_name = cast(SourceName, tool.name)
                sources_searched.append(tool_name)
                logger.info("Search tool succeeded", tool=tool.name, count=len(success_result))

        # DEDUPLICATION STEP
        original_count = len(all_evidence)
        all_evidence = deduplicate_evidence(all_evidence)
        dedup_count = original_count - len(all_evidence)

        if dedup_count > 0:
            logger.info(
                "Deduplicated evidence",
                original=original_count,
                unique=len(all_evidence),
                removed=dedup_count,
            )

        return SearchResult(
            query=query,
            evidence=all_evidence,
            sources_searched=sources_searched,
            total_found=len(all_evidence),
            errors=errors,
        )

    async def _search_with_timeout(
        self,
        tool: SearchTool,
        query: str,
        max_results: int,
    ) -> list[Evidence]:
        """Execute a single tool search with timeout."""
        try:
            return await asyncio.wait_for(
                tool.search(query, max_results),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            raise SearchError(f"{tool.name} search timed out after {self.timeout}s") from e
