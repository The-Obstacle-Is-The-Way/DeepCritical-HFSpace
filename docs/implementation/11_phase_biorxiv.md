# Phase 11 Implementation Spec: bioRxiv Preprint Integration

**Goal**: Add cutting-edge preprint search for the latest research.
**Philosophy**: "Preprints are where breakthroughs appear first."
**Prerequisite**: Phase 10 complete (ClinicalTrials.gov working)
**Estimated Time**: 2-3 hours

---

## 1. Why bioRxiv?

### Scientific Value

| Feature | Value for Drug Repurposing |
|---------|---------------------------|
| **Cutting-edge research** | 6-12 months ahead of PubMed |
| **Rapid publication** | Days, not months |
| **Free full-text** | Complete papers, not just abstracts |
| **medRxiv included** | Medical preprints via same API |
| **No API key required** | Free and open |

### The Preprint Advantage

```
Traditional Publication Timeline:
  Research → Submit → Review → Revise → Accept → Publish
  |___________________________ 6-18 months _______________|

Preprint Timeline:
  Research → Upload → Available
  |______ 1-3 days ______|
```

**For drug repurposing**: Preprints contain the newest hypotheses and evidence!

---

## 2. API Specification

### Endpoint

```
Base URL: https://api.biorxiv.org/details/[server]/[interval]/[cursor]/[format]
```

### Servers

| Server | Content |
|--------|---------|
| `biorxiv` | Biology preprints |
| `medrxiv` | Medical preprints (more relevant for us!) |

### Interval Formats

| Format | Example | Description |
|--------|---------|-------------|
| Date range | `2024-01-01/2024-12-31` | Papers between dates |
| Recent N | `50` | Most recent N papers |
| Recent N days | `30d` | Papers from last N days |

### Response Format

```json
{
  "collection": [
    {
      "doi": "10.1101/2024.01.15.123456",
      "title": "Metformin repurposing for neurodegeneration",
      "authors": "Smith, J; Jones, A",
      "date": "2024-01-15",
      "category": "neuroscience",
      "abstract": "We investigated metformin's potential..."
    }
  ],
  "messages": [{"status": "ok", "count": 100}]
}
```

### Rate Limits

- No official limit, but be respectful
- Results paginated (100 per call)
- Use cursor for pagination

### Documentation

- [bioRxiv API](https://api.biorxiv.org/)
- [medrxivr R package docs](https://docs.ropensci.org/medrxivr/)

---

## 3. Search Strategy

### Challenge: bioRxiv API Limitations

The bioRxiv API does NOT support keyword search directly. It returns papers by:
- Date range
- Recent count

### Solution: Client-Side Filtering

```python
# Strategy:
# 1. Fetch recent papers (e.g., last 90 days)
# 2. Filter by keyword matching in title/abstract
# 3. Use embeddings for semantic matching (leverage Phase 6!)
```

### Alternative: Content Search Endpoint

```
https://api.biorxiv.org/pubs/[server]/[doi_prefix]
```

For searching, we can use the publisher endpoint with filtering.

---

## 4. Data Model

### 4.1 Update Citation Source Type (`src/utils/models.py`)

```python
# After Phase 11
source: Literal["pubmed", "clinicaltrials", "biorxiv"]
```

### 4.2 Evidence from Preprints

```python
Evidence(
    content=abstract[:2000],
    citation=Citation(
        source="biorxiv",  # or "medrxiv"
        title=title,
        url=f"https://doi.org/{doi}",
        date=date,
        authors=authors.split("; ")[:5]
    ),
    relevance=0.75  # Preprints slightly lower than peer-reviewed
)
```

---

## 5. Implementation

### 5.1 bioRxiv Tool (`src/tools/biorxiv.py`)

```python
"""bioRxiv/medRxiv preprint search tool."""

import re
from datetime import datetime, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class BioRxivTool:
    """Search tool for bioRxiv and medRxiv preprints."""

    BASE_URL = "https://api.biorxiv.org/details"
    # Use medRxiv for medical/clinical content (more relevant for drug repurposing)
    DEFAULT_SERVER = "medrxiv"
    # Fetch papers from last N days
    DEFAULT_DAYS = 90

    def __init__(self, server: str = DEFAULT_SERVER, days: int = DEFAULT_DAYS):
        """
        Initialize bioRxiv tool.

        Args:
            server: "biorxiv" or "medrxiv"
            days: How many days back to search
        """
        self.server = server
        self.days = days

    @property
    def name(self) -> str:
        return "biorxiv"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search bioRxiv/medRxiv for preprints matching query.

        Note: bioRxiv API doesn't support keyword search directly.
        We fetch recent papers and filter client-side.

        Args:
            query: Search query (keywords)
            max_results: Maximum results to return

        Returns:
            List of Evidence objects from preprints
        """
        # Build date range for last N days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        interval = f"{start_date}/{end_date}"

        # Fetch recent papers
        url = f"{self.BASE_URL}/{self.server}/{interval}/0/json"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise SearchError(f"bioRxiv search failed: {e}") from e

            data = response.json()
            papers = data.get("collection", [])

            # Filter papers by query keywords
            query_terms = self._extract_terms(query)
            matching = self._filter_by_keywords(papers, query_terms, max_results)

            return [self._paper_to_evidence(paper) for paper in matching]

    def _extract_terms(self, query: str) -> list[str]:
        """Extract search terms from query."""
        # Simple tokenization, lowercase
        terms = re.findall(r'\b\w+\b', query.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'for', 'and', 'or', 'of', 'to'}
        return [t for t in terms if t not in stop_words and len(t) > 2]

    def _filter_by_keywords(
        self, papers: list[dict], terms: list[str], max_results: int
    ) -> list[dict]:
        """Filter papers that contain query terms in title or abstract."""
        scored_papers = []

        for paper in papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            text = f"{title} {abstract}"

            # Count matching terms
            matches = sum(1 for term in terms if term in text)

            if matches > 0:
                scored_papers.append((matches, paper))

        # Sort by match count (descending)
        scored_papers.sort(key=lambda x: x[0], reverse=True)

        return [paper for _, paper in scored_papers[:max_results]]

    def _paper_to_evidence(self, paper: dict) -> Evidence:
        """Convert a preprint paper to Evidence."""
        doi = paper.get("doi", "")
        title = paper.get("title", "Untitled")
        authors_str = paper.get("authors", "Unknown")
        date = paper.get("date", "Unknown")
        abstract = paper.get("abstract", "No abstract available.")
        category = paper.get("category", "")

        # Parse authors (format: "Smith, J; Jones, A")
        authors = [a.strip() for a in authors_str.split(";")][:5]

        # Note this is a preprint in the content
        content = (
            f"[PREPRINT - Not peer-reviewed] "
            f"{abstract[:1800]}... "
            f"Category: {category}."
        )

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="biorxiv",
                title=title[:500],
                url=f"https://doi.org/{doi}" if doi else f"https://www.medrxiv.org/",
                date=date,
                authors=authors,
            ),
            relevance=0.75,  # Slightly lower than peer-reviewed
        )
```

---

## 6. TDD Test Suite

### 6.1 Unit Tests (`tests/unit/tools/test_biorxiv.py`)

```python
"""Unit tests for bioRxiv tool."""

import pytest
import respx
from httpx import Response

from src.tools.biorxiv import BioRxivTool
from src.utils.models import Evidence


@pytest.fixture
def mock_biorxiv_response():
    """Mock bioRxiv API response."""
    return {
        "collection": [
            {
                "doi": "10.1101/2024.01.15.24301234",
                "title": "Metformin repurposing for Alzheimer's disease: a systematic review",
                "authors": "Smith, John; Jones, Alice; Brown, Bob",
                "date": "2024-01-15",
                "category": "neurology",
                "abstract": "Background: Metformin has shown neuroprotective effects. "
                           "We conducted a systematic review of metformin's potential "
                           "for Alzheimer's disease treatment."
            },
            {
                "doi": "10.1101/2024.01.10.24301111",
                "title": "COVID-19 vaccine efficacy study",
                "authors": "Wilson, C",
                "date": "2024-01-10",
                "category": "infectious diseases",
                "abstract": "This study evaluates COVID-19 vaccine efficacy."
            }
        ],
        "messages": [{"status": "ok", "count": 2}]
    }


class TestBioRxivTool:
    """Tests for BioRxivTool."""

    def test_tool_name(self):
        """Tool should have correct name."""
        tool = BioRxivTool()
        assert tool.name == "biorxiv"

    def test_default_server_is_medrxiv(self):
        """Default server should be medRxiv for medical relevance."""
        tool = BioRxivTool()
        assert tool.server == "medrxiv"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_evidence(self, mock_biorxiv_response):
        """Search should return Evidence objects."""
        respx.get(url__startswith="https://api.biorxiv.org/details").mock(
            return_value=Response(200, json=mock_biorxiv_response)
        )

        tool = BioRxivTool()
        results = await tool.search("metformin alzheimer", max_results=5)

        assert len(results) == 1  # Only the matching paper
        assert isinstance(results[0], Evidence)
        assert results[0].citation.source == "biorxiv"
        assert "metformin" in results[0].citation.title.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_filters_by_keywords(self, mock_biorxiv_response):
        """Search should filter papers by query keywords."""
        respx.get(url__startswith="https://api.biorxiv.org/details").mock(
            return_value=Response(200, json=mock_biorxiv_response)
        )

        tool = BioRxivTool()

        # Search for metformin - should match first paper
        results = await tool.search("metformin")
        assert len(results) == 1
        assert "metformin" in results[0].citation.title.lower()

        # Search for COVID - should match second paper
        results = await tool.search("covid vaccine")
        assert len(results) == 1
        assert "covid" in results[0].citation.title.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_marks_as_preprint(self, mock_biorxiv_response):
        """Evidence content should note it's a preprint."""
        respx.get(url__startswith="https://api.biorxiv.org/details").mock(
            return_value=Response(200, json=mock_biorxiv_response)
        )

        tool = BioRxivTool()
        results = await tool.search("metformin")

        assert "PREPRINT" in results[0].content
        assert "Not peer-reviewed" in results[0].content

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_empty_results(self):
        """Search should handle empty results gracefully."""
        respx.get(url__startswith="https://api.biorxiv.org/details").mock(
            return_value=Response(200, json={"collection": [], "messages": []})
        )

        tool = BioRxivTool()
        results = await tool.search("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_api_error(self):
        """Search should raise SearchError on API failure."""
        from src.utils.exceptions import SearchError

        respx.get(url__startswith="https://api.biorxiv.org/details").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        tool = BioRxivTool()

        with pytest.raises(SearchError):
            await tool.search("metformin")

    def test_extract_terms(self):
        """Should extract meaningful search terms."""
        tool = BioRxivTool()

        terms = tool._extract_terms("metformin for Alzheimer's disease")

        assert "metformin" in terms
        assert "alzheimer" in terms
        assert "disease" in terms
        assert "for" not in terms  # Stop word
        assert "the" not in terms  # Stop word


class TestBioRxivIntegration:
    """Integration tests (marked for separate run)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test actual API call (requires network)."""
        tool = BioRxivTool(days=30)  # Last 30 days
        results = await tool.search("diabetes", max_results=3)

        # May or may not find results depending on recent papers
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, Evidence)
            assert r.citation.source == "biorxiv"
```

---

## 7. Integration with SearchHandler

### 7.1 Final SearchHandler Configuration

```python
# examples/search_demo/run_search.py
from src.tools.biorxiv import BioRxivTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler

search_handler = SearchHandler(
    tools=[
        PubMedTool(),           # Peer-reviewed papers
        ClinicalTrialsTool(),   # Clinical trials
        BioRxivTool(),          # Preprints (cutting edge)
    ],
    timeout=30.0
)
```

### 7.2 Final Type Definition

```python
# src/utils/models.py
sources_searched: list[Literal["pubmed", "clinicaltrials", "biorxiv"]]
```

---

## 8. Definition of Done

Phase 11 is **COMPLETE** when:

- [ ] `src/tools/biorxiv.py` implemented
- [ ] Unit tests in `tests/unit/tools/test_biorxiv.py`
- [ ] Integration test marked with `@pytest.mark.integration`
- [ ] SearchHandler updated to include BioRxivTool
- [ ] Type definitions updated in models.py
- [ ] Example files updated
- [ ] All unit tests pass
- [ ] Lints pass
- [ ] Manual verification with real API

---

## 9. Verification Commands

```bash
# 1. Run unit tests
uv run pytest tests/unit/tools/test_biorxiv.py -v

# 2. Run integration test (requires network)
uv run pytest tests/unit/tools/test_biorxiv.py -v -m integration

# 3. Run full test suite
uv run pytest tests/unit/ -v

# 4. Run example with all three sources
source .env && uv run python examples/search_demo/run_search.py "metformin diabetes"
# Should show results from PubMed, ClinicalTrials.gov, AND bioRxiv/medRxiv
```

---

## 10. Value Delivered

| Before | After |
|--------|-------|
| Only published papers | Published + Preprints |
| 6-18 month lag | Near real-time research |
| Miss cutting-edge | Catch breakthroughs early |

**Demo pitch (final)**:
> "DeepBoner searches PubMed for peer-reviewed evidence, ClinicalTrials.gov for 400,000+ clinical trials, and bioRxiv/medRxiv for cutting-edge preprints - then uses LLMs to generate mechanistic hypotheses and synthesize findings into publication-quality reports."

---

## 11. Complete Source Architecture (After Phase 11)

```
User Query: "Can metformin treat Alzheimer's?"
                    |
                    v
            SearchHandler
                    |
    ┌───────────────┼───────────────┐
    |               |               |
    v               v               v
PubMedTool    ClinicalTrials   BioRxivTool
    |          Tool               |
    |               |               |
    v               v               v
"15 peer-    "3 Phase II     "2 preprints
reviewed      trials          from last
papers"       recruiting"     90 days"
    |               |               |
    └───────────────┼───────────────┘
                    |
                    v
            Evidence Pool
                    |
                    v
        EmbeddingService.deduplicate()
                    |
                    v
        HypothesisAgent → JudgeAgent → ReportAgent
                    |
                    v
        Structured Research Report
```

**This is the Gucci Banger stack.**
