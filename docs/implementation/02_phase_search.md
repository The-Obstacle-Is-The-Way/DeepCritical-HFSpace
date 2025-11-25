# Phase 2 Implementation Spec: Search Vertical Slice

**Goal**: Implement the "Eyes and Ears" of the agent — retrieving real biomedical data.
**Philosophy**: "Real data, mocked connections."
**Estimated Effort**: 3-4 hours
**Prerequisite**: Phase 1 complete

---

## 1. The Slice Definition

This slice covers:
1. **Input**: A string query (e.g., "metformin Alzheimer's disease").
2. **Process**:
   - Fetch from PubMed (E-utilities API).
   - Fetch from Web (DuckDuckGo).
   - Normalize results into `Evidence` models.
3. **Output**: A list of `Evidence` objects.

**Files**:
- `src/utils/models.py`: Data models
- `src/tools/__init__.py`: SearchTool Protocol
- `src/tools/pubmed.py`: PubMed implementation
- `src/tools/websearch.py`: DuckDuckGo implementation
- `src/tools/search_handler.py`: Orchestration

---

## 2. Models (`src/utils/models.py`)

> **Note**: All models go in `src/utils/models.py` to avoid circular imports.

```python
"""Data models for DeepCritical."""
from pydantic import BaseModel, Field
from typing import Literal


class Citation(BaseModel):
    """A citation to a source document."""

    source: Literal["pubmed", "web"] = Field(description="Where this came from")
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(description="URL to the source")
    date: str = Field(description="Publication date (YYYY-MM-DD or 'Unknown')")
    authors: list[str] = Field(default_factory=list)

    @property
    def formatted(self) -> str:
        """Format as a citation string."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        return f"{author_str} ({self.date}). {self.title}. {self.source.upper()}"


class Evidence(BaseModel):
    """A piece of evidence retrieved from search."""

    content: str = Field(min_length=1, description="The actual text content")
    citation: Citation
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score 0-1")

    class Config:
        frozen = True  # Immutable after creation


class SearchResult(BaseModel):
    """Result of a search operation."""

    query: str
    evidence: list[Evidence]
    sources_searched: list[Literal["pubmed", "web"]]
    total_found: int
    errors: list[str] = Field(default_factory=list)
```

---

## 3. Tool Protocol (`src/tools/__init__.py`)

```python
"""Search tools package."""
from typing import Protocol, List
from src.utils.models import Evidence


class SearchTool(Protocol):
    """Protocol defining the interface for all search tools."""

    @property
    def name(self) -> str:
        """Human-readable name of this tool."""
        ...

    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        """Execute a search and return evidence."""
        ...
```

---

## 4. Implementations

### 4.1 PubMed Tool (`src/tools/pubmed.py`)

> **NCBI E-utilities API**: Free, no API key required for <3 req/sec.
> - ESearch: Get PMIDs matching query
> - EFetch: Get article details by PMID

```python
"""PubMed search tool using NCBI E-utilities."""
import asyncio
import httpx
import xmltodict
from typing import List, Any
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.exceptions import SearchError, RateLimitError
from src.utils.models import Evidence, Citation

logger = structlog.get_logger()


class PubMedTool:
    """Search tool for PubMed/NCBI."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key

    def __init__(self, api_key: str | None = None):
        """Initialize PubMed tool.

        Args:
            api_key: Optional NCBI API key for higher rate limits (10 req/sec).
        """
        self.api_key = api_key
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _esearch(self, query: str, max_results: int) -> list[str]:
        """Search PubMed and return PMIDs.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of PMID strings.
        """
        await self._rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
            response.raise_for_status()

            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            logger.info("pubmed_esearch_complete", query=query, count=len(id_list))
            return id_list

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def _efetch(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch article details by PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of article dictionaries.
        """
        if not pmids:
            return []

        await self._rate_limit()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
            response.raise_for_status()

            # Parse XML response
            data = xmltodict.parse(response.text)

            # Handle single vs multiple articles
            articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
            if isinstance(articles, dict):
                articles = [articles]

            logger.info("pubmed_efetch_complete", count=len(articles))
            return articles

    def _parse_article(self, article: dict[str, Any]) -> Evidence | None:
        """Parse a PubMed article into Evidence.

        Args:
            article: Raw article dictionary from XML.

        Returns:
            Evidence object or None if parsing fails.
        """
        try:
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
                # Handle structured abstracts
                abstract = " ".join(
                    item.get("#text", str(item)) if isinstance(item, dict) else str(item)
                    for item in abstract_data
                )
            elif isinstance(abstract_data, dict):
                abstract = abstract_data.get("#text", str(abstract_data))
            else:
                abstract = str(abstract_data)

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

            # Extract date
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "Unknown")
            month = pub_date.get("Month", "")
            day = pub_date.get("Day", "")
            date_str = f"{year}-{month}-{day}".rstrip("-") if month else year

            # Build URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            if not title or not abstract:
                return None

            return Evidence(
                content=abstract[:2000],  # Truncate long abstracts
                citation=Citation(
                    source="pubmed",
                    title=title[:500],
                    url=url,
                    date=date_str,
                    authors=authors,
                ),
                relevance=0.8,  # Default high relevance for PubMed results
            )
        except Exception as e:
            logger.warning("pubmed_parse_error", error=str(e))
            return None

    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        """Execute a PubMed search and return evidence.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default 10).

        Returns:
            List of Evidence objects.

        Raises:
            SearchError: If the search fails after retries.
        """
        try:
            # Step 1: ESearch to get PMIDs
            pmids = await self._esearch(query, max_results)

            if not pmids:
                logger.info("pubmed_no_results", query=query)
                return []

            # Step 2: EFetch to get article details
            articles = await self._efetch(pmids)

            # Step 3: Parse articles into Evidence
            evidence = []
            for article in articles:
                parsed = self._parse_article(article)
                if parsed:
                    evidence.append(parsed)

            logger.info("pubmed_search_complete", query=query, results=len(evidence))
            return evidence

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"PubMed rate limit exceeded: {e}")
            raise SearchError(f"PubMed search failed: {e}")
        except Exception as e:
            raise SearchError(f"PubMed search error: {e}")
```

---

### 4.2 DuckDuckGo Tool (`src/tools/websearch.py`)

> **DuckDuckGo**: Free web search, no API key required.

```python
"""Web search tool using DuckDuckGo."""
from typing import List
import structlog
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Evidence, Citation

logger = structlog.get_logger()


class WebTool:
    """Search tool for general web search via DuckDuckGo."""

    def __init__(self):
        """Initialize web search tool."""
        pass

    @property
    def name(self) -> str:
        return "web"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def _search_sync(self, query: str, max_results: int) -> list[dict]:
        """Synchronous search wrapper (DDG library is sync).

        Args:
            query: Search query.
            max_results: Maximum results to return.

        Returns:
            List of result dictionaries.
        """
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                safesearch="moderate",
            ))
        return results

    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        """Execute a web search and return evidence.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default 10).

        Returns:
            List of Evidence objects.

        Raises:
            SearchError: If the search fails after retries.
        """
        try:
            # DuckDuckGo library is synchronous, but we wrap it
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._search_sync(query, max_results)
            )

            evidence = []
            for i, result in enumerate(results):
                title = result.get("title", "")
                url = result.get("href", result.get("link", ""))
                body = result.get("body", result.get("snippet", ""))

                if not title or not body:
                    continue

                evidence.append(Evidence(
                    content=body[:1000],
                    citation=Citation(
                        source="web",
                        title=title[:500],
                        url=url,
                        date="Unknown",
                        authors=[],
                    ),
                    relevance=max(0.5, 1.0 - (i * 0.05)),  # Decay by position
                ))

            logger.info("web_search_complete", query=query, results=len(evidence))
            return evidence

        except Exception as e:
            raise SearchError(f"Web search failed: {e}")
```

---

### 4.3 Search Handler (`src/tools/search_handler.py`)

```python
"""Search handler - orchestrates multiple search tools."""
import asyncio
from typing import List, Sequence
import structlog

from src.utils.models import Evidence, SearchResult
from src.tools import SearchTool

logger = structlog.get_logger()


class SearchHandler:
    """Orchestrates parallel searches across multiple tools."""

    def __init__(self, tools: Sequence[SearchTool]):
        """Initialize with a list of search tools.

        Args:
            tools: Sequence of SearchTool implementations.
        """
        self.tools = list(tools)

    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        """Execute search across all tools in parallel.

        Args:
            query: Search query string.
            max_results_per_tool: Max results per tool (default 10).

        Returns:
            SearchResult containing combined evidence from all tools.
        """
        errors: list[str] = []
        all_evidence: list[Evidence] = []
        sources_searched: list[str] = []

        # Run all searches in parallel
        async def run_tool(tool: SearchTool) -> tuple[str, list[Evidence], str | None]:
            """Run a single tool and capture result/error."""
            try:
                results = await tool.search(query, max_results_per_tool)
                return (tool.name, results, None)
            except Exception as e:
                logger.warning("search_tool_failed", tool=tool.name, error=str(e))
                return (tool.name, [], str(e))

        # Execute all tools concurrently
        tasks = [run_tool(tool) for tool in self.tools]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        for tool_name, evidence, error in results:
            sources_searched.append(tool_name)
            all_evidence.extend(evidence)
            if error:
                errors.append(f"{tool_name}: {error}")

        # Sort by relevance (highest first)
        all_evidence.sort(key=lambda e: e.relevance, reverse=True)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_evidence: list[Evidence] = []
        for e in all_evidence:
            if e.citation.url not in seen_urls:
                seen_urls.add(e.citation.url)
                unique_evidence.append(e)

        logger.info(
            "search_complete",
            query=query,
            total_results=len(unique_evidence),
            sources=sources_searched,
            errors=len(errors),
        )

        return SearchResult(
            query=query,
            evidence=unique_evidence,
            sources_searched=sources_searched,  # type: ignore
            total_found=len(unique_evidence),
            errors=errors,
        )
```

---

## 5. TDD Workflow

### Test File: `tests/unit/tools/test_search.py`

```python
"""Unit tests for search tools."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPubMedTool:
    """Tests for PubMedTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """PubMedTool.search should return Evidence objects."""
        from src.tools.pubmed import PubMedTool
        from src.utils.models import Evidence

        # Mock the internal methods
        tool = PubMedTool()

        mocker.patch.object(
            tool, "_esearch",
            new=AsyncMock(return_value=["12345678"])
        )
        mocker.patch.object(
            tool, "_efetch",
            new=AsyncMock(return_value=[{
                "MedlineCitation": {
                    "PMID": {"#text": "12345678"},
                    "Article": {
                        "ArticleTitle": "Test Article",
                        "Abstract": {"AbstractText": "Test abstract content."},
                        "AuthorList": {"Author": [{"LastName": "Smith", "ForeName": "John"}]},
                        "Journal": {"JournalIssue": {"PubDate": {"Year": "2024"}}}
                    }
                }
            }])
        )

        results = await tool.search("test query")

        assert len(results) == 1
        assert isinstance(results[0], Evidence)
        assert results[0].citation.source == "pubmed"
        assert "12345678" in results[0].citation.url

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, mocker):
        """PubMedTool should handle empty results gracefully."""
        from src.tools.pubmed import PubMedTool

        tool = PubMedTool()
        mocker.patch.object(tool, "_esearch", new=AsyncMock(return_value=[]))

        results = await tool.search("nonexistent query xyz123")
        assert results == []

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mocker):
        """PubMedTool should respect rate limits."""
        from src.tools.pubmed import PubMedTool
        import asyncio

        tool = PubMedTool()
        tool._last_request_time = asyncio.get_event_loop().time()

        # Mock sleep to verify it's called
        sleep_mock = mocker.patch("asyncio.sleep", new=AsyncMock())

        await tool._rate_limit()

        # Should have slept to respect rate limit
        sleep_mock.assert_called()


class TestWebTool:
    """Tests for WebTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """WebTool.search should return Evidence objects."""
        from src.tools.websearch import WebTool
        from src.utils.models import Evidence

        mock_results = [
            {"title": "Test Result", "href": "https://example.com", "body": "Test content"},
            {"title": "Another Result", "href": "https://example2.com", "body": "More content"},
        ]

        # Mock the DDGS context manager
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=None)
        mock_ddgs.text = MagicMock(return_value=mock_results)

        mocker.patch("src.tools.websearch.DDGS", return_value=mock_ddgs)

        tool = WebTool()
        results = await tool.search("test query")

        assert len(results) == 2
        assert all(isinstance(r, Evidence) for r in results)
        assert results[0].citation.source == "web"

    @pytest.mark.asyncio
    async def test_search_handles_errors(self, mocker):
        """WebTool should raise SearchError on failure."""
        from src.tools.websearch import WebTool
        from src.utils.exceptions import SearchError

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(side_effect=Exception("API error"))
        mocker.patch("src.tools.websearch.DDGS", return_value=mock_ddgs)

        tool = WebTool()

        with pytest.raises(SearchError):
            await tool.search("test query")


class TestSearchHandler:
    """Tests for SearchHandler."""

    @pytest.mark.asyncio
    async def test_execute_combines_results(self, mocker):
        """SearchHandler should combine results from all tools."""
        from src.tools.search_handler import SearchHandler
        from src.utils.models import Evidence, Citation, SearchResult

        # Create mock tools
        mock_pubmed = MagicMock()
        mock_pubmed.name = "pubmed"
        mock_pubmed.search = AsyncMock(return_value=[
            Evidence(
                content="PubMed result",
                citation=Citation(
                    source="pubmed", title="PM Article",
                    url="https://pubmed.ncbi.nlm.nih.gov/1/", date="2024"
                ),
                relevance=0.9
            )
        ])

        mock_web = MagicMock()
        mock_web.name = "web"
        mock_web.search = AsyncMock(return_value=[
            Evidence(
                content="Web result",
                citation=Citation(
                    source="web", title="Web Article",
                    url="https://example.com", date="Unknown"
                ),
                relevance=0.7
            )
        ])

        handler = SearchHandler([mock_pubmed, mock_web])
        result = await handler.execute("test query")

        assert isinstance(result, SearchResult)
        assert len(result.evidence) == 2
        assert result.total_found == 2
        assert "pubmed" in result.sources_searched
        assert "web" in result.sources_searched

    @pytest.mark.asyncio
    async def test_execute_handles_partial_failures(self, mocker):
        """SearchHandler should continue if one tool fails."""
        from src.tools.search_handler import SearchHandler
        from src.utils.models import Evidence, Citation
        from src.utils.exceptions import SearchError

        # One tool succeeds, one fails
        mock_pubmed = MagicMock()
        mock_pubmed.name = "pubmed"
        mock_pubmed.search = AsyncMock(side_effect=SearchError("PubMed down"))

        mock_web = MagicMock()
        mock_web.name = "web"
        mock_web.search = AsyncMock(return_value=[
            Evidence(
                content="Web result",
                citation=Citation(
                    source="web", title="Web Article",
                    url="https://example.com", date="Unknown"
                ),
                relevance=0.7
            )
        ])

        handler = SearchHandler([mock_pubmed, mock_web])
        result = await handler.execute("test query")

        # Should still get web results
        assert len(result.evidence) == 1
        assert len(result.errors) == 1
        assert "pubmed" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_execute_deduplicates_by_url(self, mocker):
        """SearchHandler should deduplicate results by URL."""
        from src.tools.search_handler import SearchHandler
        from src.utils.models import Evidence, Citation

        # Both tools return same URL
        evidence = Evidence(
            content="Same content",
            citation=Citation(
                source="pubmed", title="Article",
                url="https://example.com/same", date="2024"
            ),
            relevance=0.8
        )

        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.search = AsyncMock(return_value=[evidence])

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.search = AsyncMock(return_value=[evidence])

        handler = SearchHandler([mock_tool1, mock_tool2])
        result = await handler.execute("test query")

        # Should deduplicate
        assert len(result.evidence) == 1
```

---

## 6. Implementation Checklist

- [ ] Add models to `src/utils/models.py` (Citation, Evidence, SearchResult)
- [ ] Create `src/tools/__init__.py` (SearchTool Protocol)
- [ ] Implement `src/tools/pubmed.py` (complete PubMedTool class)
- [ ] Implement `src/tools/websearch.py` (complete WebTool class)
- [ ] Implement `src/tools/search_handler.py` (complete SearchHandler class)
- [ ] Write tests in `tests/unit/tools/test_search.py`
- [ ] Run `uv run pytest tests/unit/tools/ -v` — **ALL TESTS MUST PASS**
- [ ] Run `uv run ruff check src/tools` — **NO ERRORS**
- [ ] Run `uv run mypy src/tools` — **NO ERRORS**
- [ ] Commit: `git commit -m "feat: phase 2 search slice complete"`

---

## 7. Definition of Done

Phase 2 is **COMPLETE** when:

1. ✅ All unit tests in `tests/unit/tools/` pass
2. ✅ `SearchHandler` returns combined results when both tools succeed
3. ✅ Graceful degradation: if PubMed fails, WebTool results still return
4. ✅ Rate limiting is enforced (no 429 errors in integration tests)
5. ✅ Ruff and mypy pass with no errors
6. ✅ Manual REPL sanity check works:

```python
import asyncio
from src.tools.pubmed import PubMedTool
from src.tools.websearch import WebTool
from src.tools.search_handler import SearchHandler

async def test():
    handler = SearchHandler([PubMedTool(), WebTool()])
    result = await handler.execute("metformin alzheimer")
    print(f"Found {result.total_found} results")
    for e in result.evidence[:3]:
        print(f"- {e.citation.title}")

asyncio.run(test())
```

**Proceed to Phase 3 ONLY after all checkboxes are complete.**
