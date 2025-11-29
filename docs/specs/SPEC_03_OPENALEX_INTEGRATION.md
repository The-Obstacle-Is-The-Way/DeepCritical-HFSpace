# SPEC 03: OpenAlex Integration

## Priority: P1 (Feature Enhancement)

## Problem Statement

We search 3 sources (PubMed, Europe PMC, ClinicalTrials.gov) but have **no citation metrics**. We can't distinguish a highly-cited landmark paper from an obscure one. OpenAlex provides:

1. **Citation counts** - Prioritize authoritative papers
2. **Citation networks** - "Who cites whom"
3. **Concept tagging** - Hierarchical categorization
4. **Related works** - ML-powered similarity
5. **Open access links** - Direct PDF URLs

**FREE API. No key required. 209M+ works indexed.**

## Groundwork Already Done

```python
# src/utils/models.py:9
SourceName = Literal["pubmed", "clinicaltrials", "europepmc", "preprint", "openalex", "web"]

# src/utils/models.py:39-42
metadata: dict[str, Any] = Field(
    default_factory=dict,
    description="Additional metadata (e.g., cited_by_count, concepts, is_open_access)",
)
```

The infrastructure is ready. We just need to build the tool.

## OpenAlex API Reference

### Endpoint

```
GET https://api.openalex.org/works
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `search` | Full-text search across title, abstract, fulltext |
| `filter` | Constrain results (e.g., `type:article`) |
| `sort` | Order results (e.g., `cited_by_count:desc`) |
| `per_page` | Results per page (max 200) |
| `mailto` | Email for polite pool (higher rate limits) |

### Example Request

```bash
GET https://api.openalex.org/works?search=metformin%20cancer&filter=type:article&sort=cited_by_count:desc&per_page=10&mailto=deepboner@example.com
```

### Response Structure

```json
{
  "results": [
    {
      "id": "https://openalex.org/W2741809807",
      "doi": "https://doi.org/10.1234/example",
      "display_name": "Paper Title",
      "publication_year": 2024,
      "cited_by_count": 150,
      "abstract_inverted_index": {
        "word1": [0],
        "word2": [1, 5]
      },
      "concepts": [
        {"display_name": "Metformin", "score": 0.95, "level": 2}
      ],
      "authorships": [
        {"author": {"display_name": "John Smith"}}
      ],
      "open_access": {
        "is_oa": true,
        "oa_url": "https://example.com/pdf"
      },
      "best_oa_location": {
        "pdf_url": "https://example.com/paper.pdf"
      }
    }
  ]
}
```

### Abstract Reconstruction

OpenAlex stores abstracts as inverted index for compression. Must reconstruct:

```python
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
```

## Architecture

### Class Diagram

```
┌─────────────────────────────────────┐
│          SearchTool (Protocol)       │
│  ─────────────────────────────────  │
│  + name: str                         │
│  + search(query, max_results) → list[Evidence]  │
└──────────────────┬──────────────────┘
                   │ implements
┌──────────────────▼──────────────────┐
│           OpenAlexTool               │
│  ─────────────────────────────────  │
│  - BASE_URL: str                     │
│  - _polite_email: str                │
│  ─────────────────────────────────  │
│  + name → "openalex"                 │
│  + search(query, max_results) → list[Evidence]  │
│  - _reconstruct_abstract(inverted_index) → str  │
│  - _to_evidence(work) → Evidence     │
│  - _extract_authors(authorships) → list[str]    │
│  - _extract_concepts(concepts) → list[str]      │
└─────────────────────────────────────┘
```

### Integration Flow

```
SearchHandler
    │
    ├── PubMedTool
    ├── EuropePMCTool
    ├── ClinicalTrialsTool
    └── OpenAlexTool  ◄── NEW
            │
            ▼
    Evidence(
        content=abstract[:2000],
        citation=Citation(source="openalex", ...),
        metadata={
            "cited_by_count": 150,
            "concepts": ["Metformin", "Cancer"],
            "is_open_access": True,
            "pdf_url": "https://..."
        }
    )
```

## TDD Implementation Plan

### Red Phase: Write Failing Tests First

**File: `tests/unit/tools/test_openalex.py`**

```python
"""Unit tests for OpenAlex tool - TDD RED phase."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.openalex import OpenAlexTool
from src.utils.models import Evidence


# Sample OpenAlex response
SAMPLE_OPENALEX_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W12345",
            "doi": "https://doi.org/10.1234/test",
            "display_name": "Metformin in Cancer Treatment",
            "publication_year": 2024,
            "cited_by_count": 150,
            "abstract_inverted_index": {
                "Metformin": [0],
                "shows": [1],
                "promise": [2],
                "in": [3],
                "cancer": [4],
                "treatment": [5],
            },
            "concepts": [
                {"display_name": "Metformin", "score": 0.95, "level": 2},
                {"display_name": "Cancer", "score": 0.88, "level": 1},
            ],
            "authorships": [
                {"author": {"display_name": "John Smith"}},
                {"author": {"display_name": "Jane Doe"}},
            ],
            "open_access": {"is_oa": True, "oa_url": "https://example.com/oa"},
            "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
        }
    ]
}


@pytest.mark.unit
class TestOpenAlexTool:
    """Tests for OpenAlexTool."""

    @pytest.fixture
    def tool(self) -> OpenAlexTool:
        return OpenAlexTool()

    def test_tool_name(self, tool: OpenAlexTool) -> None:
        """Tool name should be 'openalex'."""
        assert tool.name == "openalex"

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool: OpenAlexTool) -> None:
        """Search should return Evidence objects."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_OPENALEX_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("metformin cancer", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert results[0].citation.source == "openalex"

    @pytest.mark.asyncio
    async def test_search_includes_citation_count(self, tool: OpenAlexTool) -> None:
        """Evidence metadata should include cited_by_count."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_OPENALEX_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("metformin cancer", max_results=5)

            assert results[0].metadata["cited_by_count"] == 150

    @pytest.mark.asyncio
    async def test_search_includes_concepts(self, tool: OpenAlexTool) -> None:
        """Evidence metadata should include concepts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_OPENALEX_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("metformin cancer", max_results=5)

            assert "Metformin" in results[0].metadata["concepts"]
            assert "Cancer" in results[0].metadata["concepts"]

    @pytest.mark.asyncio
    async def test_search_includes_open_access_info(self, tool: OpenAlexTool) -> None:
        """Evidence metadata should include open access info."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_OPENALEX_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("metformin cancer", max_results=5)

            assert results[0].metadata["is_open_access"] is True
            assert results[0].metadata["pdf_url"] == "https://example.com/paper.pdf"

    def test_reconstruct_abstract(self, tool: OpenAlexTool) -> None:
        """Abstract reconstruction from inverted index."""
        inverted_index = {
            "Hello": [0],
            "world": [1],
            "this": [2],
            "is": [3],
            "a": [4],
            "test": [5],
        }

        result = tool._reconstruct_abstract(inverted_index)

        assert result == "Hello world this is a test"

    def test_reconstruct_abstract_empty(self, tool: OpenAlexTool) -> None:
        """Handle None/empty inverted index."""
        assert tool._reconstruct_abstract(None) == ""
        assert tool._reconstruct_abstract({}) == ""

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool: OpenAlexTool) -> None:
        """Handle empty results gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("xyznonexistent123", max_results=5)

            assert results == []

    @pytest.mark.asyncio
    async def test_search_sorts_by_citations(self, tool: OpenAlexTool) -> None:
        """Verify API call requests citation-sorted results."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            await tool.search("test query", max_results=5)

            # Verify call params
            call_args = mock_instance.get.call_args
            params = call_args[1]["params"]
            assert params["sort"] == "cited_by_count:desc"

    @pytest.mark.asyncio
    async def test_search_uses_polite_pool(self, tool: OpenAlexTool) -> None:
        """Verify mailto parameter for polite pool."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            await tool.search("test query", max_results=5)

            call_args = mock_instance.get.call_args
            params = call_args[1]["params"]
            assert "mailto" in params
```

### Green Phase: Implement to Pass Tests

**File: `src/tools/openalex.py`**

```python
"""OpenAlex search tool - citation-aware scholarly search."""

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
            "filter": "type:article",  # Only peer-reviewed articles
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

        # Prepend citation badge to content
        citation_badge = f"[Cited by {cited_by_count}] " if cited_by_count > 0 else ""
        content = f"{citation_badge}{abstract[:1900]}"

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="openalex",
                title=title[:500],
                url=url,
                date=str(year),
                authors=authors,
            ),
            relevance=min(1.0, cited_by_count / 1000),  # Normalize to 0-1
            metadata={
                "cited_by_count": cited_by_count,
                "concepts": concepts,
                "is_open_access": is_oa,
                "pdf_url": pdf_url,
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
```

### Refactor Phase: Clean Integration

**Update: `src/tools/__init__.py`**

```python
"""Search tools package."""

from src.tools.base import SearchTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.openalex import OpenAlexTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler

__all__ = [
    "ClinicalTrialsTool",
    "EuropePMCTool",
    "OpenAlexTool",
    "PubMedTool",
    "SearchHandler",
    "SearchTool",
]
```

## Test Matrix

| Test | What It Validates | Priority |
|------|------------------|----------|
| `test_tool_name` | Returns "openalex" | P0 |
| `test_search_returns_evidence` | Returns `list[Evidence]` | P0 |
| `test_search_includes_citation_count` | `metadata["cited_by_count"]` populated | P0 |
| `test_search_includes_concepts` | `metadata["concepts"]` populated | P0 |
| `test_search_includes_open_access_info` | `metadata["is_open_access"]`, `pdf_url` | P1 |
| `test_reconstruct_abstract` | Inverted index → text | P0 |
| `test_reconstruct_abstract_empty` | Handle None/empty | P0 |
| `test_search_empty_results` | Return `[]` for no matches | P0 |
| `test_search_sorts_by_citations` | API param `sort=cited_by_count:desc` | P0 |
| `test_search_uses_polite_pool` | API param `mailto` present | P1 |

## Integration Test

```python
@pytest.mark.integration
class TestOpenAlexIntegration:
    """Integration tests with real OpenAlex API."""

    @pytest.mark.asyncio
    async def test_real_api_returns_results(self) -> None:
        """Test actual API returns relevant results."""
        tool = OpenAlexTool()
        results = await tool.search("metformin cancer treatment", max_results=3)

        assert len(results) > 0
        # Should have citation counts
        assert results[0].metadata["cited_by_count"] >= 0
        # Should have concepts
        assert len(results[0].metadata["concepts"]) > 0
```

## Acceptance Criteria

- [ ] `OpenAlexTool` implements `SearchTool` Protocol
- [ ] `search()` returns `list[Evidence]` sorted by citation count
- [ ] `metadata["cited_by_count"]` populated for each result
- [ ] `metadata["concepts"]` contains top 5 concept names
- [ ] `metadata["is_open_access"]` and `pdf_url` populated
- [ ] Abstract correctly reconstructed from inverted index
- [ ] Empty results handled gracefully
- [ ] Tool exported from `src/tools/__init__.py`
- [ ] All unit tests pass with mocked API
- [ ] Integration test passes with real API
- [ ] `make check` passes (lint + type + test)

## Files to Create/Modify

### Create

1. `src/tools/openalex.py` - Main tool class
2. `tests/unit/tools/test_openalex.py` - Unit tests

### Modify

1. `src/tools/__init__.py` - Add export
2. `src/orchestrator.py` - Add to tool list (if using default tools)

## SOLID Principles Applied

| Principle | Application |
|-----------|------------|
| **S**ingle Responsibility | `OpenAlexTool` only handles OpenAlex API |
| **O**pen/Closed | New tool without modifying existing ones |
| **L**iskov Substitution | Implements `SearchTool` Protocol, interchangeable |
| **I**nterface Segregation | Uses minimal `SearchTool` interface |
| **D**ependency Inversion | `SearchHandler` depends on Protocol, not concrete class |

## DRY Applied

- Reuses `Evidence`, `Citation`, `SearchError` from `src/utils/models.py`
- Follows identical patterns as `PubMedTool`, `EuropePMCTool`
- Same retry decorator pattern as other tools

## Related Issues

- #51: feat: Add OpenAlex as 4th data source

## Effort Estimate

- Tests (TDD Red): 30 min
- Implementation (Green): 1 hour
- Refactor + Integration: 30 min
- **Total: ~2 hours**
