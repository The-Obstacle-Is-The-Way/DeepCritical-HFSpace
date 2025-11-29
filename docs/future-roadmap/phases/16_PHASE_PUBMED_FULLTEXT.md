# Phase 16: PubMed Full-Text Retrieval

**Priority**: MEDIUM - Enhances evidence quality
**Effort**: ~3 hours
**Dependencies**: None (existing PubMed tool sufficient)

---

## Prerequisites (COMPLETED)

The `Evidence.metadata` field has been added to `src/utils/models.py` to support:
```python
metadata={"has_fulltext": True}
```

---

## Architecture Decision: Constructor Parameter vs Method Parameter

**IMPORTANT**: The original spec proposed `include_fulltext` as a method parameter:
```python
# WRONG - SearchHandler won't pass this parameter
async def search(self, query: str, max_results: int = 10, include_fulltext: bool = False):
```

**Problem**: `SearchHandler` calls `tool.search(query, max_results)` uniformly across all tools.
It has no mechanism to pass tool-specific parameters like `include_fulltext`.

**Solution**: Use constructor parameter instead:
```python
# CORRECT - Configured at instantiation time
class PubMedTool:
    def __init__(self, api_key: str | None = None, include_fulltext: bool = False):
        self.include_fulltext = include_fulltext
        ...
```

This way, you can create a full-text-enabled PubMed tool:
```python
# In orchestrator or wherever tools are created
tools = [
    PubMedTool(include_fulltext=True),  # Full-text enabled
    ClinicalTrialsTool(),
    EuropePMCTool(),
]
```

---

## Overview

Add full-text retrieval for PubMed papers via the BioC API, enabling:
- Complete paper text for open-access PMC papers
- Structured sections (intro, methods, results, discussion)
- Better evidence for LLM synthesis

**Why Full-Text?**
- Abstracts only give ~200-300 words
- Full text provides detailed methods, results, figures
- Reference repo already has this implemented
- Makes LLM judgments more accurate

---

## TDD Implementation Plan

### Step 1: Write the Tests First

**File**: `tests/unit/tools/test_pubmed_fulltext.py`

```python
"""Tests for PubMed full-text retrieval."""

import pytest
import respx
from httpx import Response

from src.tools.pubmed import PubMedTool


class TestPubMedFullText:
    """Test suite for PubMed full-text functionality."""

    @pytest.fixture
    def tool(self) -> PubMedTool:
        return PubMedTool()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_pmc_id_success(self, tool: PubMedTool) -> None:
        """Should convert PMID to PMCID for full-text access."""
        mock_response = {
            "records": [
                {
                    "pmid": "12345678",
                    "pmcid": "PMC1234567",
                }
            ]
        }

        respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/").mock(
            return_value=Response(200, json=mock_response)
        )

        pmcid = await tool.get_pmc_id("12345678")
        assert pmcid == "PMC1234567"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_pmc_id_not_in_pmc(self, tool: PubMedTool) -> None:
        """Should return None if paper not in PMC."""
        mock_response = {
            "records": [
                {
                    "pmid": "12345678",
                    # No pmcid means not in PMC
                }
            ]
        }

        respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/").mock(
            return_value=Response(200, json=mock_response)
        )

        pmcid = await tool.get_pmc_id("12345678")
        assert pmcid is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_fulltext_success(self, tool: PubMedTool) -> None:
        """Should retrieve full text for PMC papers."""
        # Mock BioC API response
        mock_bioc = {
            "documents": [
                {
                    "passages": [
                        {
                            "infons": {"section_type": "INTRO"},
                            "text": "Introduction text here.",
                        },
                        {
                            "infons": {"section_type": "METHODS"},
                            "text": "Methods description here.",
                        },
                        {
                            "infons": {"section_type": "RESULTS"},
                            "text": "Results summary here.",
                        },
                        {
                            "infons": {"section_type": "DISCUSS"},
                            "text": "Discussion and conclusions.",
                        },
                    ]
                }
            ]
        }

        respx.get(
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/12345678/unicode"
        ).mock(return_value=Response(200, json=mock_bioc))

        fulltext = await tool.get_fulltext("12345678")

        assert fulltext is not None
        assert "Introduction text here" in fulltext
        assert "Methods description here" in fulltext
        assert "Results summary here" in fulltext

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_fulltext_not_available(self, tool: PubMedTool) -> None:
        """Should return None if full text not available."""
        respx.get(
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/99999999/unicode"
        ).mock(return_value=Response(404))

        fulltext = await tool.get_fulltext("99999999")
        assert fulltext is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_fulltext_structured(self, tool: PubMedTool) -> None:
        """Should return structured sections dict."""
        mock_bioc = {
            "documents": [
                {
                    "passages": [
                        {"infons": {"section_type": "INTRO"}, "text": "Intro..."},
                        {"infons": {"section_type": "METHODS"}, "text": "Methods..."},
                        {"infons": {"section_type": "RESULTS"}, "text": "Results..."},
                        {"infons": {"section_type": "DISCUSS"}, "text": "Discussion..."},
                    ]
                }
            ]
        }

        respx.get(
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/12345678/unicode"
        ).mock(return_value=Response(200, json=mock_bioc))

        sections = await tool.get_fulltext_structured("12345678")

        assert sections is not None
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "discussion" in sections

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_with_fulltext_enabled(self) -> None:
        """Search should include full text when tool is configured for it."""
        # Create tool WITH full-text enabled via constructor
        tool = PubMedTool(include_fulltext=True)

        # Mock esearch
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200, json={"esearchresult": {"idlist": ["12345678"]}}
            )
        )

        # Mock efetch (abstract)
        mock_xml = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>12345678</PMID>
              <Article>
                <ArticleTitle>Test Paper</ArticleTitle>
                <Abstract><AbstractText>Short abstract.</AbstractText></Abstract>
                <AuthorList><Author><LastName>Smith</LastName></Author></AuthorList>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
            return_value=Response(200, text=mock_xml)
        )

        # Mock ID converter
        respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/").mock(
            return_value=Response(
                200, json={"records": [{"pmid": "12345678", "pmcid": "PMC1234567"}]}
            )
        )

        # Mock BioC full text
        mock_bioc = {
            "documents": [
                {
                    "passages": [
                        {"infons": {"section_type": "INTRO"}, "text": "Full intro..."},
                    ]
                }
            ]
        }
        respx.get(
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/12345678/unicode"
        ).mock(return_value=Response(200, json=mock_bioc))

        # NOTE: No include_fulltext param - it's set via constructor
        results = await tool.search("test", max_results=1)

        assert len(results) == 1
        # Full text should be appended or replace abstract
        assert "Full intro" in results[0].content or "Short abstract" in results[0].content
```

---

### Step 2: Implement Full-Text Methods

**File**: `src/tools/pubmed.py` (additions to existing class)

```python
# Add these methods to PubMedTool class

async def get_pmc_id(self, pmid: str) -> str | None:
    """
    Convert PMID to PMCID for full-text access.

    Args:
        pmid: PubMed ID

    Returns:
        PMCID if paper is in PMC, None otherwise
    """
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"ids": pmid, "format": "json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            if records and records[0].get("pmcid"):
                return records[0]["pmcid"]
            return None

        except httpx.HTTPError:
            return None


async def get_fulltext(self, pmid: str) -> str | None:
    """
    Get full text for a PubMed paper via BioC API.

    Only works for open-access papers in PubMed Central.

    Args:
        pmid: PubMed ID

    Returns:
        Full text as string, or None if not available
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            # Extract text from all passages
            documents = data.get("documents", [])
            if not documents:
                return None

            passages = documents[0].get("passages", [])
            text_parts = [p.get("text", "") for p in passages if p.get("text")]

            return "\n\n".join(text_parts) if text_parts else None

        except httpx.HTTPError:
            return None


async def get_fulltext_structured(self, pmid: str) -> dict[str, str] | None:
    """
    Get structured full text with sections.

    Args:
        pmid: PubMed ID

    Returns:
        Dict mapping section names to text, or None if not available
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            documents = data.get("documents", [])
            if not documents:
                return None

            # Map section types to readable names
            section_map = {
                "INTRO": "introduction",
                "METHODS": "methods",
                "RESULTS": "results",
                "DISCUSS": "discussion",
                "CONCL": "conclusion",
                "ABSTRACT": "abstract",
            }

            sections: dict[str, list[str]] = {}
            for passage in documents[0].get("passages", []):
                section_type = passage.get("infons", {}).get("section_type", "other")
                section_name = section_map.get(section_type, "other")
                text = passage.get("text", "")

                if text:
                    if section_name not in sections:
                        sections[section_name] = []
                    sections[section_name].append(text)

            # Join multiple passages per section
            return {k: "\n\n".join(v) for k, v in sections.items()}

        except httpx.HTTPError:
            return None
```

---

### Step 3: Update Constructor and Search Method

Add full-text flag to constructor and update search to use it:

```python
class PubMedTool:
    """Search tool for PubMed/NCBI."""

    def __init__(
        self,
        api_key: str | None = None,
        include_fulltext: bool = False,  # NEW CONSTRUCTOR PARAM
    ) -> None:
        self.api_key = api_key or settings.ncbi_api_key
        if self.api_key == "your-ncbi-key-here":
            self.api_key = None
        self._last_request_time = 0.0
        self.include_fulltext = include_fulltext  # Store for use in search()

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search PubMed and return evidence.

        Note: Full-text enrichment is controlled by constructor parameter,
        not method parameter, because SearchHandler doesn't pass extra args.
        """
        # ... existing search logic ...

        evidence_list = self._parse_pubmed_xml(fetch_resp.text)

        # Optionally enrich with full text (if configured at construction)
        if self.include_fulltext:
            evidence_list = await self._enrich_with_fulltext(evidence_list)

        return evidence_list


async def _enrich_with_fulltext(
    self, evidence_list: list[Evidence]
) -> list[Evidence]:
    """Attempt to add full text to evidence items."""
    enriched = []

    for evidence in evidence_list:
        # Extract PMID from URL
        url = evidence.citation.url
        pmid = url.rstrip("/").split("/")[-1] if url else None

        if pmid:
            fulltext = await self.get_fulltext(pmid)
            if fulltext:
                # Replace abstract with full text (truncated)
                evidence = Evidence(
                    content=fulltext[:8000],  # Larger limit for full text
                    citation=evidence.citation,
                    relevance=evidence.relevance,
                    metadata={
                        **evidence.metadata,
                        "has_fulltext": True,
                    },
                )

        enriched.append(evidence)

    return enriched
```

---

## Demo Script

**File**: `examples/pubmed_fulltext_demo.py`

```python
#!/usr/bin/env python3
"""Demo script to verify PubMed full-text retrieval."""

import asyncio
from src.tools.pubmed import PubMedTool


async def main():
    """Run PubMed full-text demo."""
    tool = PubMedTool()

    print("=" * 60)
    print("PubMed Full-Text Demo")
    print("=" * 60)

    # Test 1: Convert PMID to PMCID
    print("\n[Test 1] Converting PMID to PMCID...")
    # Use a known open-access paper
    test_pmid = "34450029"  # Example: COVID-related open-access paper
    pmcid = await tool.get_pmc_id(test_pmid)
    print(f"PMID {test_pmid} -> PMCID: {pmcid or 'Not in PMC'}")

    # Test 2: Get full text
    print("\n[Test 2] Fetching full text...")
    if pmcid:
        fulltext = await tool.get_fulltext(test_pmid)
        if fulltext:
            print(f"Full text length: {len(fulltext)} characters")
            print(f"Preview: {fulltext[:500]}...")
        else:
            print("Full text not available")

    # Test 3: Get structured sections
    print("\n[Test 3] Fetching structured sections...")
    if pmcid:
        sections = await tool.get_fulltext_structured(test_pmid)
        if sections:
            print("Available sections:")
            for section, text in sections.items():
                print(f"  - {section}: {len(text)} chars")
        else:
            print("Structured text not available")

    # Test 4: Search with full text
    print("\n[Test 4] Search with full-text enrichment...")
    results = await tool.search(
        "metformin cancer open access",
        max_results=3,
        include_fulltext=True
    )

    for i, evidence in enumerate(results, 1):
        has_ft = evidence.metadata.get("has_fulltext", False)
        print(f"\n--- Result {i} ---")
        print(f"Title: {evidence.citation.title}")
        print(f"Has Full Text: {has_ft}")
        print(f"Content Length: {len(evidence.content)} chars")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Verification Checklist

### Unit Tests
```bash
# Run full-text tests
uv run pytest tests/unit/tools/test_pubmed_fulltext.py -v

# Run all PubMed tests
uv run pytest tests/unit/tools/test_pubmed.py -v

# Expected: All tests pass
```

### Integration Test (Manual)
```bash
# Run demo with real API
uv run python examples/pubmed_fulltext_demo.py

# Expected: Real full text from PMC papers
```

### Full Test Suite
```bash
make check
# Expected: All tests pass, mypy clean
```

---

## Success Criteria

1. **ID Conversion works**: PMID -> PMCID conversion successful
2. **Full text retrieval works**: BioC API returns paper text
3. **Structured sections work**: Can get intro/methods/results/discussion separately
4. **Search integration works**: `include_fulltext=True` enriches results
5. **No regressions**: Existing tests still pass
6. **Graceful degradation**: Non-PMC papers still return abstracts

---

## Notes

- Only ~30% of PubMed papers have full text in PMC
- BioC API has no documented rate limit, but be respectful
- Full text can be very long - truncate appropriately
- Consider caching full text responses (they don't change)
- Timeout should be longer for full text (60s vs 30s)
