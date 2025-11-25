"""PubMed search tool using NCBI E-utilities."""

import asyncio
from typing import Any

import httpx
import xmltodict
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.exceptions import RateLimitError, SearchError
from src.utils.models import Citation, Evidence


class PubMedTool:
    """Search tool for PubMed/NCBI."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key
    HTTP_TOO_MANY_REQUESTS = 429

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or getattr(settings, "ncbi_api_key", None)
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "pubmed"

    async def _rate_limit(self) -> None:
        """Enforce NCBI rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        """Build request params with optional API key."""
        params = {**kwargs, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(  # type: ignore[misc]
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search PubMed and return evidence.

        1. ESearch: Get PMIDs matching query
        2. EFetch: Get abstracts for those PMIDs
        3. Parse and return Evidence objects
        """
        await self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Search for PMIDs
            search_params = self._build_params(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
            )

            try:
                search_resp = await client.get(
                    f"{self.BASE_URL}/esearch.fcgi",
                    params=search_params,
                )
                search_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == self.HTTP_TOO_MANY_REQUESTS:
                    raise RateLimitError("PubMed rate limit exceeded") from e
                raise SearchError(f"PubMed search failed: {e}") from e

            search_data = search_resp.json()
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Step 2: Fetch abstracts
            await self._rate_limit()
            fetch_params = self._build_params(
                db="pubmed",
                id=",".join(pmids),
                rettype="abstract",
            )
            # Use XML for fetch (more reliable parsing)
            fetch_params["retmode"] = "xml"

            fetch_resp = await client.get(
                f"{self.BASE_URL}/efetch.fcgi",
                params=fetch_params,
            )
            fetch_resp.raise_for_status()

            # Step 3: Parse XML to Evidence
            return self._parse_pubmed_xml(fetch_resp.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[Evidence]:
        """Parse PubMed XML into Evidence objects."""
        try:
            data = xmltodict.parse(xml_text)
        except Exception as e:
            raise SearchError(f"Failed to parse PubMed XML: {e}") from e

        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])

        # Handle single article (xmltodict returns dict instead of list)
        if isinstance(articles, dict):
            articles = [articles]

        evidence_list = []
        for article in articles:
            try:
                evidence = self._article_to_evidence(article)
                if evidence:
                    evidence_list.append(evidence)
            except Exception:
                continue  # Skip malformed articles

        return evidence_list

    def _article_to_evidence(self, article: dict[str, Any]) -> Evidence | None:
        """Convert a single PubMed article to Evidence."""
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})

        # Extract PMID
        pmid = medline.get("PMID", {})
        if isinstance(pmid, dict):
            pmid = pmid.get("#text", "")

        # Extract title
        title = article_data.get("ArticleTitle", "")
        if isinstance(title, dict):
            title = title.get("#text", str(title))

        # Extract abstract
        abstract_data = article_data.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abstract_data, list):
            abstract = " ".join(
                item.get("#text", str(item)) if isinstance(item, dict) else str(item)
                for item in abstract_data
            )
        elif isinstance(abstract_data, dict):
            abstract = abstract_data.get("#text", str(abstract_data))
        else:
            abstract = str(abstract_data)

        if not abstract or not title:
            return None

        # Extract date
        pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "Unknown")
        month = pub_date.get("Month", "01")
        day = pub_date.get("Day", "01")
        date_str = f"{year}-{month}-{day}" if year != "Unknown" else "Unknown"

        # Extract authors
        author_list = article_data.get("AuthorList", {}).get("Author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        authors = []
        for author in author_list[:5]:  # Limit to 5 authors
            last = author.get("LastName", "")
            first = author.get("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())

        return Evidence(
            content=abstract[:2000],  # Truncate long abstracts
            citation=Citation(
                source="pubmed",
                title=title[:500],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                date=date_str,
                authors=authors,
            ),
        )
