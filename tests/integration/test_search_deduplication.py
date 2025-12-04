import pytest

from src.tools.europepmc import EuropePMCTool
from src.tools.openalex import OpenAlexTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler, extract_paper_id


@pytest.mark.integration
@pytest.mark.slow
async def test_real_search_deduplicates() -> None:
    """Integration test: Real search should deduplicate PubMed/Europe PMC."""

    # Initialize tools
    # Note: PubMedTool handles missing API key gracefully (lower rate limit)
    handler = SearchHandler(
        tools=[PubMedTool(), EuropePMCTool(), OpenAlexTool()],
        timeout=30.0,
    )

    # Execute search
    # "sildenafil erectile dysfunction" is a well-indexed topic likely to appear in all sources
    result = await handler.execute("sildenafil erectile dysfunction", max_results_per_tool=5)

    # Checks
    # 1. Total results should be less than sum of max_results (5 * 3 = 15) if deduplication works
    # (There's a high chance of overlap between PubMed, EuropePMC, and OpenAlex)
    assert result.total_found > 0, "Search should return some results"

    # Note: We can't strictly assert result.total_found < 15 because it's theoretically possible
    # (though unlikely) to get 15 unique papers. But for this query, overlap is expected.
    # A better check is to verify uniqueness explicitly.

    # 2. Verify no duplicate IDs in the returned evidence
    # extract_paper_id filter already excludes falsy values (including None)
    paper_ids = [extract_paper_id(e) for e in result.evidence if extract_paper_id(e)]

    # Check for duplicates
    unique_ids = set(paper_ids)
    assert len(paper_ids) == len(unique_ids), (
        f"Duplicate IDs found: {[x for x in paper_ids if paper_ids.count(x) > 1]}"
    )
