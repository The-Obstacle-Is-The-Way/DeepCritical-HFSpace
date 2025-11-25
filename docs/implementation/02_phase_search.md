# Phase 2 Implementation Spec: Search Vertical Slice

**Goal**: Implement the "Eyes and Ears" of the agent — retrieving real biomedical data.
**Philosophy**: "Real data, mocked connections."
**Prerequisite**: Phase 1 complete (all tests passing)

---

## 1. The Slice Definition

This slice covers:
1. **Input**: A string query (e.g., "metformin Alzheimer's disease").
2. **Process**:
   - Fetch from PubMed (E-utilities API).
   - Fetch from Web (DuckDuckGo).
   - Normalize results into `Evidence` models.
3. **Output**: A list of `Evidence` objects.

**Files to Create**:
- `src/utils/models.py` - Pydantic models (Evidence, Citation, SearchResult)
- `src/tools/pubmed.py` - PubMed E-utilities tool
- `src/tools/websearch.py` - DuckDuckGo search tool
- `src/tools/search_handler.py` - Orchestrates multiple tools
- `src/tools/__init__.py` - Exports

---

## 2. PubMed E-utilities API Reference

**Base URL**: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

### Key Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `esearch.fcgi` | Search for article IDs | `?db=pubmed&term=metformin+alzheimer&retmax=10` |
| `efetch.fcgi` | Fetch article details | `?db=pubmed&id=12345,67890&rettype=abstract&retmode=xml` |

### Rate Limiting (CRITICAL!)

NCBI **requires** rate limiting:
- **Without API key**: 3 requests/second
- **With API key**: 10 requests/second

Get a free API key: https://www.ncbi.nlm.nih.gov/account/settings/

```python
# Add to .env
NCBI_API_KEY=your-key-here  # Optional but recommended
```

### Example Search Flow

```
1. esearch: "metformin alzheimer" → [PMID: 12345, 67890, ...]
2. efetch: PMIDs → Full abstracts/metadata
3. Parse XML → Evidence objects
```

---

## 3. Models (`src/utils/models.py`)

```python
"""Data models for the Search feature."""
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

## 4. Tool Protocol (`src/tools/pubmed.py` and `src/tools/websearch.py`)

### The Interface (Protocol) - Add to `src/tools/__init__.py`

```python
"""Search tools package."""
from typing import Protocol, List

# Import implementations
from src.tools.pubmed import PubMedTool
from src.tools.websearch import WebTool
from src.tools.search_handler import SearchHandler

# Re-export
__all__ = ["SearchTool", "PubMedTool", "WebTool", "SearchHandler"]


class SearchTool(Protocol):
    """Protocol defining the interface for all search tools."""

    @property
    def name(self) -> str:
        """Human-readable name of this tool."""
        ...

    async def search(self, query: str, max_results: int = 10) -> List["Evidence"]:
        """
        Execute a search and return evidence.

        Args:
            query: The search query string
            max_results: Maximum number of results to return

        Returns:
            List of Evidence objects

        Raises:
            SearchError: If the search fails
            RateLimitError: If we hit rate limits
        """
        ...
```

### PubMed Tool Implementation (`src/tools/pubmed.py`)

```python
"""PubMed search tool using NCBI E-utilities."""
import asyncio
import httpx
import xmltodict
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.exceptions import SearchError, RateLimitError
from src.utils.models import Evidence, Citation


class PubMedTool:
    """Search tool for PubMed/NCBI."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key

    def __init__(self, api_key: str | None = None):
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

    def _build_params(self, **kwargs) -> dict:
        """Build request params with optional API key."""
        params = {**kwargs, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
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
                if e.response.status_code == 429:
                    raise RateLimitError("PubMed rate limit exceeded")
                raise SearchError(f"PubMed search failed: {e}")

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

    def _parse_pubmed_xml(self, xml_text: str) -> List[Evidence]:
        """Parse PubMed XML into Evidence objects."""
        try:
            data = xmltodict.parse(xml_text)
        except Exception as e:
            raise SearchError(f"Failed to parse PubMed XML: {e}")

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

    def _article_to_evidence(self, article: dict) -> Evidence | None:
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
```

### DuckDuckGo Tool Implementation (`src/tools/websearch.py`)

```python
"""Web search tool using DuckDuckGo."""
from typing import List
from duckduckgo_search import DDGS

from src.utils.exceptions import SearchError
from src.utils.models import Evidence, Citation


class WebTool:
    """Search tool for general web search via DuckDuckGo."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "web"

    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        """
        Search DuckDuckGo and return evidence.

        Note: duckduckgo-search is synchronous, so we run it in executor.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_search(query, max_results),
            )
            return results
        except Exception as e:
            raise SearchError(f"Web search failed: {e}")

    def _sync_search(self, query: str, max_results: int) -> List[Evidence]:
        """Synchronous search implementation."""
        evidence_list = []

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        for result in results:
            evidence_list.append(
                Evidence(
                    content=result.get("body", "")[:1000],
                    citation=Citation(
                        source="web",
                        title=result.get("title", "Unknown")[:500],
                        url=result.get("href", ""),
                        date="Unknown",
                        authors=[],
                    ),
                )
            )

        return evidence_list
```

---

## 5. Search Handler (`src/tools/search_handler.py`)

The handler orchestrates multiple tools using the **Scatter-Gather** pattern.

```python
"""Search handler - orchestrates multiple search tools."""
import asyncio
from typing import List, Protocol
import structlog

from src.utils.exceptions import SearchError
from src.utils.models import Evidence, SearchResult

logger = structlog.get_logger()


class SearchTool(Protocol):
    """Protocol defining the interface for all search tools."""

    @property
    def name(self) -> str:
        ...

    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        ...


def flatten(nested: List[List[Evidence]]) -> List[Evidence]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested for item in sublist]


class SearchHandler:
    """Orchestrates parallel searches across multiple tools."""

    def __init__(self, tools: List[SearchTool], timeout: float = 30.0):
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
            self._search_with_timeout(tool, query, max_results_per_tool)
            for tool in self.tools
        ]

        # Gather results (don't fail if one tool fails)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_evidence: List[Evidence] = []
        sources_searched: List[str] = []
        errors: List[str] = []

        for tool, result in zip(self.tools, results):
            if isinstance(result, Exception):
                errors.append(f"{tool.name}: {str(result)}")
                logger.warning("Search tool failed", tool=tool.name, error=str(result))
            else:
                all_evidence.extend(result)
                sources_searched.append(tool.name)
                logger.info("Search tool succeeded", tool=tool.name, count=len(result))

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
    ) -> List[Evidence]:
        """Execute a single tool search with timeout."""
        try:
            return await asyncio.wait_for(
                tool.search(query, max_results),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise SearchError(f"{tool.name} search timed out after {self.timeout}s")
```

---

## 6. TDD Workflow

### Test File: `tests/unit/tools/test_pubmed.py`

```python
"""Unit tests for PubMed tool."""
import pytest
from unittest.mock import AsyncMock, MagicMock


# Sample PubMed XML response for mocking
SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Metformin in Alzheimer's Disease: A Systematic Review</ArticleTitle>
                <Abstract>
                    <AbstractText>Metformin shows neuroprotective properties...</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                </AuthorList>
                <Journal>
                    <JournalIssue>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>01</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedTool:
    """Tests for PubMedTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """PubMedTool should return Evidence objects from search."""
        from src.tools.pubmed import PubMedTool

        # Mock the HTTP responses
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": ["12345678"]}
        }
        mock_search_response.raise_for_status = MagicMock()

        mock_fetch_response = MagicMock()
        mock_fetch_response.text = SAMPLE_PUBMED_XML
        mock_fetch_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_fetch_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Act
        tool = PubMedTool()
        results = await tool.search("metformin alzheimer")

        # Assert
        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Metformin" in results[0].citation.title
        assert "12345678" in results[0].citation.url

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mocker):
        """PubMedTool should return empty list when no results."""
        from src.tools.pubmed import PubMedTool

        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        results = await tool.search("xyznonexistentquery123")

        assert results == []

    def test_parse_pubmed_xml(self):
        """PubMedTool should correctly parse XML."""
        from src.tools.pubmed import PubMedTool

        tool = PubMedTool()
        results = tool._parse_pubmed_xml(SAMPLE_PUBMED_XML)

        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Smith John" in results[0].citation.authors
```

### Test File: `tests/unit/tools/test_websearch.py`

```python
"""Unit tests for WebTool."""
import pytest
from unittest.mock import MagicMock


class TestWebTool:
    """Tests for WebTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """WebTool should return Evidence objects from search."""
        from src.tools.websearch import WebTool

        mock_results = [
            {
                "title": "Drug Repurposing Article",
                "href": "https://example.com/article",
                "body": "Some content about drug repurposing...",
            }
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=None)
        mock_ddgs.text = MagicMock(return_value=mock_results)

        mocker.patch("src.tools.websearch.DDGS", return_value=mock_ddgs)

        tool = WebTool()
        results = await tool.search("drug repurposing")

        assert len(results) == 1
        assert results[0].citation.source == "web"
        assert "Drug Repurposing" in results[0].citation.title
```

### Test File: `tests/unit/tools/test_search_handler.py`

```python
"""Unit tests for SearchHandler."""
import pytest
from unittest.mock import AsyncMock

from src.utils.models import Evidence, Citation
from src.utils.exceptions import SearchError


class TestSearchHandler:
    """Tests for SearchHandler."""

    @pytest.mark.asyncio
    async def test_execute_aggregates_results(self):
        """SearchHandler should aggregate results from all tools."""
        from src.tools.search_handler import SearchHandler

        # Create mock tools
        mock_tool_1 = AsyncMock()
        mock_tool_1.name = "mock1"
        mock_tool_1.search = AsyncMock(return_value=[
            Evidence(
                content="Result 1",
                citation=Citation(source="pubmed", title="T1", url="u1", date="2024"),
            )
        ])

        mock_tool_2 = AsyncMock()
        mock_tool_2.name = "mock2"
        mock_tool_2.search = AsyncMock(return_value=[
            Evidence(
                content="Result 2",
                citation=Citation(source="web", title="T2", url="u2", date="2024"),
            )
        ])

        handler = SearchHandler(tools=[mock_tool_1, mock_tool_2])
        result = await handler.execute("test query")

        assert result.total_found == 2
        assert "mock1" in result.sources_searched
        assert "mock2" in result.sources_searched
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_handles_tool_failure(self):
        """SearchHandler should continue if one tool fails."""
        from src.tools.search_handler import SearchHandler

        mock_tool_ok = AsyncMock()
        mock_tool_ok.name = "ok_tool"
        mock_tool_ok.search = AsyncMock(return_value=[
            Evidence(
                content="Good result",
                citation=Citation(source="pubmed", title="T", url="u", date="2024"),
            )
        ])

        mock_tool_fail = AsyncMock()
        mock_tool_fail.name = "fail_tool"
        mock_tool_fail.search = AsyncMock(side_effect=SearchError("API down"))

        handler = SearchHandler(tools=[mock_tool_ok, mock_tool_fail])
        result = await handler.execute("test")

        assert result.total_found == 1
        assert "ok_tool" in result.sources_searched
        assert len(result.errors) == 1
        assert "fail_tool" in result.errors[0]
```

---

## 7. Integration Test (Optional, Real API)

```python
# tests/integration/test_pubmed_live.py
"""Integration tests that hit real APIs (run manually)."""
import pytest


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_pubmed_live_search():
    """Test real PubMed search (requires network)."""
    from src.tools.pubmed import PubMedTool

    tool = PubMedTool()
    results = await tool.search("metformin diabetes", max_results=3)

    assert len(results) > 0
    assert results[0].citation.source == "pubmed"
    assert "pubmed.ncbi.nlm.nih.gov" in results[0].citation.url


# Run with: uv run pytest tests/integration -m integration
```

---

## 8. Implementation Checklist

- [ ] Create `src/utils/models.py` with all Pydantic models (Evidence, Citation, SearchResult)
- [ ] Create `src/tools/__init__.py` with SearchTool Protocol and exports
- [ ] Implement `src/tools/pubmed.py` with PubMedTool class
- [ ] Implement `src/tools/websearch.py` with WebTool class
- [ ] Create `src/tools/search_handler.py` with SearchHandler class
- [ ] Write tests in `tests/unit/tools/test_pubmed.py`
- [ ] Write tests in `tests/unit/tools/test_websearch.py`
- [ ] Write tests in `tests/unit/tools/test_search_handler.py`
- [ ] Run `uv run pytest tests/unit/tools/ -v` — **ALL TESTS MUST PASS**
- [ ] (Optional) Run integration test: `uv run pytest -m integration`
- [ ] Commit: `git commit -m "feat: phase 2 search slice complete"`

---

## 9. Definition of Done

Phase 2 is **COMPLETE** when:

1. All unit tests pass: `uv run pytest tests/unit/tools/ -v`
2. `SearchHandler` can execute with both tools
3. Graceful degradation: if PubMed fails, WebTool results still return
4. Rate limiting is enforced (verify no 429 errors)
5. Can run this in Python REPL:

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
