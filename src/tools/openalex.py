"""OpenAlex search tool - citation-aware scholarly search."""

import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class OpenAlexTool:
    """
    Search OpenAlex for scholarly works with citation metrics.

    OpenAlex indexes 209M+ works and provides:
    - Citation counts (prioritize influential papers)
    - Concept tagging (hierarchical classification)
    - Open access links (direct PDF URLs)
    - Related works (ML-powered similarity)

    API Docs: https://docs.openalex.org
    Rate Limits: Polite pool with mailto = 100k/day
    """

    BASE_URL = "https://api.openalex.org/works"
    POLITE_EMAIL = "deepboner-research@proton.me"

    @property
    def name(self) -> str:
        return "openalex"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search OpenAlex, sorted by citation count.

        Args:
            query: Search terms
            max_results: Maximum results to return

        Returns:
            List of Evidence objects with citation metadata
        """
        params: dict[str, str | int] = {
            "search": query,
            "filter": "type:article,has_abstract:true",  # Only articles with abstracts
            "sort": "cited_by_count:desc",  # Most cited first
            "per_page": min(max_results, 100),
            "mailto": self.POLITE_EMAIL,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                works = data.get("results", [])

                return [self._to_evidence(work) for work in works[:max_results]]

            except httpx.HTTPStatusError as e:
                raise SearchError(f"OpenAlex API error: {e}") from e
            except httpx.RequestError as e:
                raise SearchError(f"OpenAlex connection failed: {e}") from e

    def _to_evidence(self, work: dict[str, Any]) -> Evidence:
        """Convert OpenAlex work to Evidence with rich metadata."""
        # Extract basic fields
        title = work.get("display_name", "Untitled")
        doi = work.get("doi", "")
        year = work.get("publication_year", "Unknown")
        cited_by_count = work.get("cited_by_count", 0)

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            # Should be caught by filter=has_abstract:true, but defensive coding
            abstract = f"[No abstract available. Cited by {cited_by_count} works.]"

        # Extract authors (limit to 5)
        authors = self._extract_authors(work.get("authorships", []))

        # Extract concepts (top 5 by score)
        concepts = self._extract_concepts(work.get("concepts", []))

        # Open access info
        oa_info = work.get("open_access", {})
        is_oa = oa_info.get("is_oa", False)

        # Get PDF URL (prefer best_oa_location)
        best_oa = work.get("best_oa_location", {})
        pdf_url = best_oa.get("pdf_url") if best_oa else None

        # Build URL
        if doi:
            url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        else:
            openalex_id = work.get("id", "")
            url = openalex_id if openalex_id else "https://openalex.org"

        # NEW: Extract PMID from ids object for deduplication
        ids_obj = work.get("ids", {})
        pmid_url = ids_obj.get("pmid")  # "https://pubmed.ncbi.nlm.nih.gov/29456894"
        pmid = None
        if pmid_url and isinstance(pmid_url, str) and "pubmed.ncbi.nlm.nih.gov" in pmid_url:
            # Extract numeric PMID from URL
            pmid_match = re.search(r"/(\d+)/?$", pmid_url)
            if pmid_match:
                pmid = pmid_match.group(1)

        # Prepend citation badge to content
        citation_badge = f"[Cited by {cited_by_count}] " if cited_by_count > 0 else ""
        content = f"{citation_badge}{abstract[:1900]}"

        # Calculate relevance: normalized citation count (capped at 1.0 for 100 citations)
        # 100 citations is a very strong signal in most fields.
        relevance = min(1.0, cited_by_count / 100.0)

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="openalex",
                title=title[:500],
                url=url,
                date=str(year),
                authors=authors,
            ),
            relevance=relevance,
            metadata={
                "cited_by_count": cited_by_count,
                "concepts": concepts,
                "is_open_access": is_oa,
                "pdf_url": pdf_url,
                "pmid": pmid,  # NEW: Store PMID for deduplication
            },
        )

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]] | None) -> str:
        """Rebuild abstract from {"word": [positions]} format."""
        if not inverted_index:
            return ""

        position_word: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        if not position_word:
            return ""

        max_pos = max(position_word.keys())
        return " ".join(position_word.get(i, "") for i in range(max_pos + 1))

    def _extract_authors(self, authorships: list[dict[str, Any]]) -> list[str]:
        """Extract author names from authorships array."""
        authors = []
        for authorship in authorships[:5]:
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)
        return authors

    def _extract_concepts(self, concepts: list[dict[str, Any]]) -> list[str]:
        """Extract concept names, sorted by score."""
        sorted_concepts = sorted(concepts, key=lambda c: c.get("score", 0), reverse=True)
        return [c.get("display_name", "") for c in sorted_concepts[:5] if c.get("display_name")]
