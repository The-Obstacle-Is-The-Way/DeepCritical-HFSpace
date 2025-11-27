# Phase 15: OpenAlex Integration

**Priority**: HIGH - Biggest bang for buck
**Effort**: ~2-3 hours
**Dependencies**: None (existing codebase patterns sufficient)

---

## Prerequisites (COMPLETED)

The following model changes have been implemented to support this integration:

1. **`SourceName` Literal Updated** (`src/utils/models.py:9`)
   ```python
   SourceName = Literal["pubmed", "clinicaltrials", "europepmc", "preprint", "openalex"]
   ```
   - Without this, `source="openalex"` would fail Pydantic validation

2. **`Evidence.metadata` Field Added** (`src/utils/models.py:39-42`)
   ```python
   metadata: dict[str, Any] = Field(
       default_factory=dict,
       description="Additional metadata (e.g., cited_by_count, concepts, is_open_access)",
   )
   ```
   - Required for storing `cited_by_count`, `concepts`, etc.
   - Model is still frozen - metadata must be passed at construction time

3. **`__init__.py` Exports Updated** (`src/tools/__init__.py`)
   - All tools are now exported: `ClinicalTrialsTool`, `EuropePMCTool`, `PubMedTool`
   - OpenAlexTool should be added here after implementation

---

## Overview

Add OpenAlex as a 4th data source for comprehensive scholarly data including:
- Citation networks (who cites whom)
- Concept tagging (hierarchical topic classification)
- Author disambiguation
- 209M+ works indexed

**Why OpenAlex?**
- Free, no API key required
- Already implemented in reference repo
- Provides citation data we don't have
- Aggregates PubMed + preprints + more

---

## TDD Implementation Plan

### Step 1: Write the Tests First

**File**: `tests/unit/tools/test_openalex.py`

```python
"""Tests for OpenAlex search tool."""

import pytest
import respx
from httpx import Response

from src.tools.openalex import OpenAlexTool
from src.utils.models import Evidence


class TestOpenAlexTool:
    """Test suite for OpenAlex search functionality."""

    @pytest.fixture
    def tool(self) -> OpenAlexTool:
        return OpenAlexTool()

    def test_name_property(self, tool: OpenAlexTool) -> None:
        """Tool should identify itself as 'openalex'."""
        assert tool.name == "openalex"

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool: OpenAlexTool) -> None:
        """Search should return list of Evidence objects."""
        mock_response = {
            "results": [
                {
                    "id": "W2741809807",
                    "title": "Metformin and cancer: A systematic review",
                    "publication_year": 2023,
                    "cited_by_count": 45,
                    "type": "article",
                    "is_oa": True,
                    "primary_location": {
                        "source": {"display_name": "Nature Medicine"},
                        "landing_page_url": "https://doi.org/10.1038/example",
                        "pdf_url": None,
                    },
                    "abstract_inverted_index": {
                        "Metformin": [0],
                        "shows": [1],
                        "anticancer": [2],
                        "effects": [3],
                    },
                    "concepts": [
                        {"display_name": "Medicine", "score": 0.95},
                        {"display_name": "Oncology", "score": 0.88},
                    ],
                    "authorships": [
                        {
                            "author": {"display_name": "John Smith"},
                            "institutions": [{"display_name": "Harvard"}],
                        }
                    ],
                }
            ]
        }

        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(200, json=mock_response)
        )

        results = await tool.search("metformin cancer", max_results=10)

        assert len(results) == 1
        assert isinstance(results[0], Evidence)
        assert "Metformin and cancer" in results[0].citation.title
        assert results[0].citation.source == "openalex"

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool: OpenAlexTool) -> None:
        """Search with no results should return empty list."""
        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(200, json={"results": []})
        )

        results = await tool.search("xyznonexistentquery123")
        assert results == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_handles_missing_abstract(self, tool: OpenAlexTool) -> None:
        """Tool should handle papers without abstracts."""
        mock_response = {
            "results": [
                {
                    "id": "W123",
                    "title": "Paper without abstract",
                    "publication_year": 2023,
                    "cited_by_count": 10,
                    "type": "article",
                    "is_oa": False,
                    "primary_location": {
                        "source": {"display_name": "Journal"},
                        "landing_page_url": "https://example.com",
                    },
                    "abstract_inverted_index": None,
                    "concepts": [],
                    "authorships": [],
                }
            ]
        }

        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(200, json=mock_response)
        )

        results = await tool.search("test query")
        assert len(results) == 1
        assert results[0].content == ""  # No abstract

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_extracts_citation_count(self, tool: OpenAlexTool) -> None:
        """Citation count should be in metadata."""
        mock_response = {
            "results": [
                {
                    "id": "W456",
                    "title": "Highly cited paper",
                    "publication_year": 2020,
                    "cited_by_count": 500,
                    "type": "article",
                    "is_oa": True,
                    "primary_location": {
                        "source": {"display_name": "Science"},
                        "landing_page_url": "https://example.com",
                    },
                    "abstract_inverted_index": {"Test": [0]},
                    "concepts": [],
                    "authorships": [],
                }
            ]
        }

        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(200, json=mock_response)
        )

        results = await tool.search("highly cited")
        assert results[0].metadata["cited_by_count"] == 500

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_extracts_concepts(self, tool: OpenAlexTool) -> None:
        """Concepts should be extracted for semantic discovery."""
        mock_response = {
            "results": [
                {
                    "id": "W789",
                    "title": "Drug repurposing study",
                    "publication_year": 2023,
                    "cited_by_count": 25,
                    "type": "article",
                    "is_oa": True,
                    "primary_location": {
                        "source": {"display_name": "PLOS ONE"},
                        "landing_page_url": "https://example.com",
                    },
                    "abstract_inverted_index": {"Drug": [0], "repurposing": [1]},
                    "concepts": [
                        {"display_name": "Pharmacology", "score": 0.92},
                        {"display_name": "Drug Discovery", "score": 0.85},
                        {"display_name": "Medicine", "score": 0.80},
                    ],
                    "authorships": [],
                }
            ]
        }

        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(200, json=mock_response)
        )

        results = await tool.search("drug repurposing")
        assert "Pharmacology" in results[0].metadata["concepts"]
        assert "Drug Discovery" in results[0].metadata["concepts"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_api_error_raises_search_error(
        self, tool: OpenAlexTool
    ) -> None:
        """API errors should raise SearchError."""
        from src.utils.exceptions import SearchError

        respx.get("https://api.openalex.org/works").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        with pytest.raises(SearchError):
            await tool.search("test query")

    def test_reconstruct_abstract(self, tool: OpenAlexTool) -> None:
        """Test abstract reconstruction from inverted index."""
        inverted_index = {
            "Metformin": [0, 5],
            "is": [1],
            "a": [2],
            "diabetes": [3],
            "drug": [4],
            "effective": [6],
        }
        abstract = tool._reconstruct_abstract(inverted_index)
        assert abstract == "Metformin is a diabetes drug Metformin effective"
```

---

### Step 2: Create the Implementation

**File**: `src/tools/openalex.py`

```python
"""OpenAlex search tool for comprehensive scholarly data."""

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class OpenAlexTool:
    """
    Search OpenAlex for scholarly works with rich metadata.

    OpenAlex provides:
    - 209M+ scholarly works
    - Citation counts and networks
    - Concept tagging (hierarchical)
    - Author disambiguation
    - Open access links

    API Docs: https://docs.openalex.org/
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, email: str | None = None) -> None:
        """
        Initialize OpenAlex tool.

        Args:
            email: Optional email for polite pool (faster responses)
        """
        self.email = email or "deepcritical@example.com"

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
        Search OpenAlex for scholarly works.

        Args:
            query: Search terms
            max_results: Maximum results to return (max 200 per request)

        Returns:
            List of Evidence objects with citation metadata

        Raises:
            SearchError: If API request fails
        """
        params = {
            "search": query,
            "filter": "type:article",  # Only peer-reviewed articles
            "sort": "cited_by_count:desc",  # Most cited first
            "per_page": min(max_results, 200),
            "mailto": self.email,  # Polite pool for faster responses
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                return [self._to_evidence(work) for work in results[:max_results]]

            except httpx.HTTPStatusError as e:
                raise SearchError(f"OpenAlex API error: {e}") from e
            except httpx.RequestError as e:
                raise SearchError(f"OpenAlex connection failed: {e}") from e

    def _to_evidence(self, work: dict[str, Any]) -> Evidence:
        """Convert OpenAlex work to Evidence object."""
        title = work.get("title", "Untitled")
        pub_year = work.get("publication_year", "Unknown")
        cited_by = work.get("cited_by_count", 0)
        is_oa = work.get("is_oa", False)

        # Reconstruct abstract from inverted index
        abstract_index = work.get("abstract_inverted_index")
        abstract = self._reconstruct_abstract(abstract_index) if abstract_index else ""

        # Extract concepts (top 5)
        concepts = [
            c.get("display_name", "")
            for c in work.get("concepts", [])[:5]
            if c.get("display_name")
        ]

        # Extract authors (top 5)
        authorships = work.get("authorships", [])
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in authorships[:5]
            if a.get("author", {}).get("display_name")
        ]

        # Get URL
        primary_loc = work.get("primary_location") or {}
        url = primary_loc.get("landing_page_url", "")
        if not url:
            # Fallback to OpenAlex page
            work_id = work.get("id", "").replace("https://openalex.org/", "")
            url = f"https://openalex.org/{work_id}"

        return Evidence(
            content=abstract[:2000],
            citation=Citation(
                source="openalex",
                title=title[:500],
                url=url,
                date=str(pub_year),
                authors=authors,
            ),
            relevance=min(0.9, 0.5 + (cited_by / 1000)),  # Boost by citations
            metadata={
                "cited_by_count": cited_by,
                "is_open_access": is_oa,
                "concepts": concepts,
                "pdf_url": primary_loc.get("pdf_url"),
            },
        )

    def _reconstruct_abstract(
        self, inverted_index: dict[str, list[int]]
    ) -> str:
        """
        Reconstruct abstract from OpenAlex inverted index format.

        OpenAlex stores abstracts as {"word": [position1, position2, ...]}.
        This rebuilds the original text.
        """
        if not inverted_index:
            return ""

        # Build position -> word mapping
        position_word: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        # Reconstruct in order
        if not position_word:
            return ""

        max_pos = max(position_word.keys())
        words = [position_word.get(i, "") for i in range(max_pos + 1)]
        return " ".join(w for w in words if w)
```

---

### Step 3: Register in Search Handler

**File**: `src/tools/search_handler.py` (add to imports and tool list)

```python
# Add import
from src.tools.openalex import OpenAlexTool

# Add to _create_tools method
def _create_tools(self) -> list[SearchTool]:
    return [
        PubMedTool(),
        ClinicalTrialsTool(),
        EuropePMCTool(),
        OpenAlexTool(),  # NEW
    ]
```

---

### Step 4: Update `__init__.py`

**File**: `src/tools/__init__.py`

```python
from src.tools.openalex import OpenAlexTool

__all__ = [
    "PubMedTool",
    "ClinicalTrialsTool",
    "EuropePMCTool",
    "OpenAlexTool",  # NEW
    # ...
]
```

---

## Demo Script

**File**: `examples/openalex_demo.py`

```python
#!/usr/bin/env python3
"""Demo script to verify OpenAlex integration."""

import asyncio
from src.tools.openalex import OpenAlexTool


async def main():
    """Run OpenAlex search demo."""
    tool = OpenAlexTool()

    print("=" * 60)
    print("OpenAlex Integration Demo")
    print("=" * 60)

    # Test 1: Basic drug repurposing search
    print("\n[Test 1] Searching for 'metformin cancer drug repurposing'...")
    results = await tool.search("metformin cancer drug repurposing", max_results=5)

    for i, evidence in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Title: {evidence.citation.title}")
        print(f"Year: {evidence.citation.date}")
        print(f"Citations: {evidence.metadata.get('cited_by_count', 'N/A')}")
        print(f"Concepts: {', '.join(evidence.metadata.get('concepts', []))}")
        print(f"Open Access: {evidence.metadata.get('is_open_access', False)}")
        print(f"URL: {evidence.citation.url}")
        if evidence.content:
            print(f"Abstract: {evidence.content[:200]}...")

    # Test 2: High-impact papers
    print("\n" + "=" * 60)
    print("[Test 2] Finding highly-cited papers on 'long COVID treatment'...")
    results = await tool.search("long COVID treatment", max_results=3)

    for evidence in results:
        print(f"\n- {evidence.citation.title}")
        print(f"  Citations: {evidence.metadata.get('cited_by_count', 0)}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Verification Checklist

### Unit Tests
```bash
# Run just OpenAlex tests
uv run pytest tests/unit/tools/test_openalex.py -v

# Expected: All tests pass
```

### Integration Test (Manual)
```bash
# Run demo script with real API
uv run python examples/openalex_demo.py

# Expected: Real results from OpenAlex API
```

### Full Test Suite
```bash
# Ensure nothing broke
make check

# Expected: All 110+ tests pass, mypy clean
```

---

## Success Criteria

1. **Unit tests pass**: All mocked tests in `test_openalex.py` pass
2. **Integration works**: Demo script returns real results
3. **No regressions**: `make check` passes completely
4. **SearchHandler integration**: OpenAlex appears in search results alongside other sources
5. **Citation metadata**: Results include `cited_by_count`, `concepts`, `is_open_access`

---

## Future Enhancements (P2)

Once basic integration works:

1. **Citation Network Queries**
   ```python
   # Get papers citing a specific work
   async def get_citing_works(self, work_id: str) -> list[Evidence]:
       params = {"filter": f"cites:{work_id}"}
       ...
   ```

2. **Concept-Based Search**
   ```python
   # Search by OpenAlex concept ID
   async def search_by_concept(self, concept_id: str) -> list[Evidence]:
       params = {"filter": f"concepts.id:{concept_id}"}
       ...
   ```

3. **Author Tracking**
   ```python
   # Find all works by an author
   async def search_by_author(self, author_id: str) -> list[Evidence]:
       params = {"filter": f"authorships.author.id:{author_id}"}
       ...
   ```

---

## Notes

- OpenAlex is **very generous** with rate limits (no documented hard limit)
- Adding `mailto` parameter gives priority access (polite pool)
- Abstract is stored as inverted index - must reconstruct
- Citation count is a good proxy for paper quality/impact
- Consider caching responses for repeated queries
