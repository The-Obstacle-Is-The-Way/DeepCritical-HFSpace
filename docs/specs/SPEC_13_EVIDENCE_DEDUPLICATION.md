# SPEC_13: Evidence Deduplication in SearchHandler

**Status**: Draft (Validated via API Documentation Review)
**Priority**: P1
**GitHub Issue**: #94
**Estimated Effort**: Medium (~100 lines of code, includes OpenAlex metadata extraction)
**Last Updated**: 2025-11-30

---

## Problem Statement

DeepBoner searches 4 sources in parallel:
1. **PubMed** - NCBI's biomedical literature database
2. **ClinicalTrials.gov** - Clinical trial registry
3. **Europe PMC** - Indexes **ALL of PubMed/MEDLINE** plus preprints
4. **OpenAlex** - Indexes **250M+ works** including most of PubMed

**Result**: The same paper appears 2-3 times in evidence because:
- Europe PMC includes 100% of PubMed content
- OpenAlex includes ~90% of PubMed content

### Impact

| Metric | Current | After Fix |
|--------|---------|-----------|
| Duplicate evidence | 30-50% | <5% |
| Token waste | High | Low |
| Judge confusion | Yes | No |
| Report redundancy | Yes | No |

---

## API Documentation Review (2025-11-30)

### OpenAlex Work Object - `ids` Field

**Source**: [OpenAlex Work Object Docs](https://docs.openalex.org/api-entities/works/work-object)

OpenAlex returns PMIDs in the `ids` field as **full URLs** (validated via live API testing 2025-11-30):
```json
{
  "ids": {
    "openalex": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.7717/peerj.4375",
    "pmid": "https://pubmed.ncbi.nlm.nih.gov/29456894"
  }
}
```

**Live API Test (2025-11-30):**
```bash
curl "https://api.openalex.org/works?filter=ids.pmid:29456894&select=id,ids"
# Returns: "pmid": "https://pubmed.ncbi.nlm.nih.gov/29456894" (always URL format, never just number)
```

**Current Issue**: Our `openalex.py` does NOT extract `ids.pmid` - it only uses DOI.
**Fix Required**: Extract PMID from OpenAlex response URL and store numeric PMID in `Evidence.metadata`.

### Europe PMC URL Patterns

**Source**: [Europe PMC Help](https://europepmc.org/help)

Four URL patterns exist:
- `/article/MED/{PMID}` - PubMed/MEDLINE records
- `/article/PMC/{PMCID}` - PubMed Central full text
- `/article/PPR/{PPRID}` - Preprints (medRxiv, bioRxiv, etc.)
- `/article/PAT/{PatentID}` - Patents (e.g., `PAT/WO8601415`, `PAT/EP1234567`)

**Live API Test (2025-11-30):**
```bash
curl "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=SRC:PAT&pageSize=1&format=json"
# Returns: source: "PAT", id: "WO8601415", pmid: None
```

**Current Behavior**: Our `europepmc.py:92-95` already uses PubMed URL when PMID exists:
```python
if doi:
    url = f"https://doi.org/{doi}"
elif result.get("pmid"):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/"  # ← Good!
else:
    url = f"https://europepmc.org/article/{source_db}/{result.get('id', '')}"
```

**Implications**:
1. For MEDLINE records with PMIDs → PubMed URL used → deduplication works
2. For PMC/PPR/PAT records without PMIDs → Europe PMC URL used → treated as unique (correct behavior)

### ClinicalTrials.gov URL Patterns

Two URL formats exist:
- Modern: `/study/NCT12345678` (current API v2)
- Legacy: `/ct2/show/NCT12345678` (deprecated but may exist in old data)

---

## Current Code

```python
# src/tools/search_handler.py:51-62
for tool, result in zip(self.tools, results, strict=True):
    if isinstance(result, Exception):
        errors.append(f"{tool.name}: {result!s}")
    else:
        success_result = cast(list[Evidence], result)
        all_evidence.extend(success_result)  # ← NO DEDUPLICATION
```

---

## Proposed Solution (Two Parts)

### Part 1: Extract PMID from OpenAlex (REQUIRED for cross-source dedup)

```python
# src/tools/openalex.py - Update _to_evidence() method

def _to_evidence(self, work: dict[str, Any]) -> Evidence:
    """Convert OpenAlex work to Evidence with rich metadata."""
    # ... existing code ...

    # NEW: Extract PMID from ids object for deduplication
    ids_obj = work.get("ids", {})
    pmid_url = ids_obj.get("pmid")  # "https://pubmed.ncbi.nlm.nih.gov/29456894"
    pmid = None
    if pmid_url and "pubmed.ncbi.nlm.nih.gov" in pmid_url:
        # Extract numeric PMID from URL
        import re
        pmid_match = re.search(r'/(\d+)/?$', pmid_url)
        if pmid_match:
            pmid = pmid_match.group(1)

    # ... rest of existing code ...

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
```

### Part 2: Enhanced Deduplication Function

```python
# src/tools/search_handler.py

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.models import Evidence


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
    pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', url)
    if pmid_match:
        return f"PMID:{pmid_match.group(1)}"

    # Europe PMC MED pattern (same as PMID)
    epmc_med_match = re.search(r'europepmc\.org/article/MED/(\d+)', url)
    if epmc_med_match:
        return f"PMID:{epmc_med_match.group(1)}"

    # Europe PMC PMC pattern (PubMed Central ID - different from PMID!)
    epmc_pmc_match = re.search(r'europepmc\.org/article/PMC/(PMC\d+)', url)
    if epmc_pmc_match:
        return f"PMCID:{epmc_pmc_match.group(1)}"

    # Europe PMC PPR pattern (Preprint ID - unique per preprint)
    epmc_ppr_match = re.search(r'europepmc\.org/article/PPR/(PPR\d+)', url)
    if epmc_ppr_match:
        return f"PPRID:{epmc_ppr_match.group(1)}"

    # Europe PMC PAT pattern (Patent ID - e.g., WO8601415, EP1234567)
    epmc_pat_match = re.search(r'europepmc\.org/article/PAT/([A-Z]{2}\d+)', url)
    if epmc_pat_match:
        return f"PATID:{epmc_pat_match.group(1)}"

    # DOI pattern (normalize trailing slash/characters)
    doi_match = re.search(r'doi\.org/(10\.\d+/[^\s\]>]+)', url)
    if doi_match:
        doi = doi_match.group(1).rstrip('/')
        return f"DOI:{doi}"

    # OpenAlex ID pattern (fallback if no PMID in metadata)
    openalex_match = re.search(r'openalex\.org/(W\d+)', url)
    if openalex_match:
        return f"OAID:{openalex_match.group(1)}"

    # ClinicalTrials NCT ID (modern format)
    nct_match = re.search(r'clinicaltrials\.gov/study/(NCT\d+)', url)
    if nct_match:
        return f"NCT:{nct_match.group(1)}"

    # ClinicalTrials NCT ID (legacy format)
    nct_legacy_match = re.search(r'clinicaltrials\.gov/ct2/show/(NCT\d+)', url)
    if nct_legacy_match:
        return f"NCT:{nct_legacy_match.group(1)}"

    return None


def deduplicate_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
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
        evidence_list,
        key=lambda e: source_priority.get(e.citation.source, 99)
    )

    for evidence in sorted_evidence:
        paper_id = extract_paper_id(evidence.citation.url)

        if paper_id is None:
            # Can't identify - keep it (conservative)
            unique.append(evidence)
            continue

        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique.append(evidence)

    return unique
```

### 2. Integrate into SearchHandler.execute()

```python
# src/tools/search_handler.py:execute() - AFTER gathering results

# ... existing code to collect all_evidence ...

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
    evidence=all_evidence,  # Now deduplicated
    sources_searched=sources_searched,
    total_found=len(all_evidence),
    errors=errors,
)
```

---

## Test Plan

### Unit Tests (`tests/unit/tools/test_search_handler.py`)

```python
import pytest
from src.tools.search_handler import extract_paper_id, deduplicate_evidence
from src.utils.models import Citation, Evidence


def _make_evidence(source: str, url: str, metadata: dict | None = None) -> Evidence:
    """Helper to create Evidence objects for testing."""
    return Evidence(
        content="Test content",
        citation=Citation(
            source=source,
            title="Test",
            url=url,
            date="2024",
            authors=[],
        ),
        metadata=metadata,
    )


class TestExtractPaperId:
    """Tests for paper ID extraction from Evidence objects."""

    def test_extracts_pubmed_id(self) -> None:
        evidence = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        assert extract_paper_id(evidence) == "PMID:12345678"

    def test_extracts_europepmc_med_id(self) -> None:
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/MED/12345678")
        assert extract_paper_id(evidence) == "PMID:12345678"

    def test_extracts_europepmc_pmc_id(self) -> None:
        """Europe PMC PMC articles have different ID format."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PMC/PMC7654321")
        assert extract_paper_id(evidence) == "PMCID:PMC7654321"

    def test_extracts_europepmc_ppr_id(self) -> None:
        """Europe PMC preprints have PPR IDs."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PPR/PPR123456")
        assert extract_paper_id(evidence) == "PPRID:PPR123456"

    def test_extracts_europepmc_pat_id(self) -> None:
        """Europe PMC patents have PAT IDs (e.g., WO8601415, EP1234567)."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PAT/WO8601415")
        assert extract_paper_id(evidence) == "PATID:WO8601415"

    def test_extracts_europepmc_pat_id_eu_format(self) -> None:
        """European patent format should also work."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PAT/EP1234567")
        assert extract_paper_id(evidence) == "PATID:EP1234567"

    def test_extracts_doi(self) -> None:
        evidence = _make_evidence("pubmed", "https://doi.org/10.1038/nature12345")
        assert extract_paper_id(evidence) == "DOI:10.1038/nature12345"

    def test_extracts_doi_with_trailing_slash(self) -> None:
        """DOIs should be normalized (trailing slash removed)."""
        evidence = _make_evidence("pubmed", "https://doi.org/10.1038/nature12345/")
        assert extract_paper_id(evidence) == "DOI:10.1038/nature12345"

    def test_extracts_openalex_id_from_url(self) -> None:
        """OpenAlex ID from URL (fallback when no PMID in metadata)."""
        evidence = _make_evidence("openalex", "https://openalex.org/W1234567890")
        assert extract_paper_id(evidence) == "OAID:W1234567890"

    def test_extracts_openalex_pmid_from_metadata(self) -> None:
        """OpenAlex PMID from metadata takes priority over URL."""
        evidence = _make_evidence(
            "openalex",
            "https://openalex.org/W1234567890",
            metadata={"pmid": "98765432"},
        )
        assert extract_paper_id(evidence) == "PMID:98765432"

    def test_extracts_nct_id_modern(self) -> None:
        evidence = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT12345678")
        assert extract_paper_id(evidence) == "NCT:NCT12345678"

    def test_extracts_nct_id_legacy(self) -> None:
        """Legacy ClinicalTrials.gov URL format should also work."""
        evidence = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/ct2/show/NCT12345678")
        assert extract_paper_id(evidence) == "NCT:NCT12345678"

    def test_returns_none_for_unknown_url(self) -> None:
        evidence = _make_evidence("unknown", "https://example.com/unknown")
        assert extract_paper_id(evidence) is None


class TestDeduplicateEvidence:
    """Tests for evidence deduplication."""

    def test_removes_pubmed_europepmc_duplicate(self) -> None:
        """Same paper from PubMed and Europe PMC should dedupe to PubMed."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        europepmc = _make_evidence("europepmc", "https://europepmc.org/article/MED/12345678")

        result = deduplicate_evidence([pubmed, europepmc])

        assert len(result) == 1
        assert result[0].citation.source == "pubmed"

    def test_removes_pubmed_openalex_duplicate_via_metadata(self) -> None:
        """OpenAlex with PMID in metadata should dedupe against PubMed."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        openalex = _make_evidence(
            "openalex",
            "https://openalex.org/W9999999",
            metadata={"pmid": "12345678", "cited_by_count": 100},
        )

        result = deduplicate_evidence([pubmed, openalex])

        assert len(result) == 1
        assert result[0].citation.source == "pubmed"

    def test_preserves_openalex_without_pmid(self) -> None:
        """OpenAlex papers without PMID should NOT be deduplicated."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        openalex_no_pmid = _make_evidence(
            "openalex",
            "https://openalex.org/W9999999",
            metadata={"cited_by_count": 100},  # No pmid key
        )

        result = deduplicate_evidence([pubmed, openalex_no_pmid])

        assert len(result) == 2  # Both preserved (different IDs)

    def test_preserves_unique_evidence(self) -> None:
        """Different papers should not be deduplicated."""
        e1 = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/11111111/")
        e2 = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/22222222/")

        result = deduplicate_evidence([e1, e2])

        assert len(result) == 2

    def test_keeps_unidentifiable_evidence(self) -> None:
        """Evidence with unrecognized URLs should be preserved."""
        unknown = _make_evidence("unknown", "https://example.com/paper/123")

        result = deduplicate_evidence([unknown])

        assert len(result) == 1

    def test_clinicaltrials_unique_per_nct(self) -> None:
        """ClinicalTrials entries have unique NCT IDs."""
        trial1 = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT11111111")
        trial2 = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT22222222")

        result = deduplicate_evidence([trial1, trial2])

        assert len(result) == 2

    def test_preprints_preserved_separately(self) -> None:
        """Preprints (PPR IDs) should not dedupe against peer-reviewed papers."""
        peer_reviewed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        preprint = _make_evidence("europepmc", "https://europepmc.org/article/PPR/PPR999999")

        result = deduplicate_evidence([peer_reviewed, preprint])

        assert len(result) == 2  # Both preserved (different ID types)
```

### Integration Test

```python
@pytest.mark.integration
async def test_real_search_deduplicates() -> None:
    """Integration test: Real search should deduplicate PubMed/Europe PMC."""
    from src.tools.pubmed import PubMedTool
    from src.tools.europepmc import EuropePMCTool
    from src.tools.openalex import OpenAlexTool
    from src.tools.search_handler import SearchHandler, extract_paper_id

    handler = SearchHandler(
        tools=[PubMedTool(), EuropePMCTool(), OpenAlexTool()],
        timeout=30.0,
    )

    result = await handler.execute("sildenafil erectile dysfunction", max_results_per_tool=5)

    # Should have fewer results than 3x max_results due to deduplication
    assert result.total_found < 15

    # Check no duplicate paper IDs in final result
    paper_ids = [
        extract_paper_id(e)
        for e in result.evidence
        if extract_paper_id(e)
    ]
    assert len(paper_ids) == len(set(paper_ids)), f"Duplicate IDs found: {paper_ids}"
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/tools/openalex.py` | Extract PMID from `ids.pmid` and store in `Evidence.metadata` |
| `src/tools/search_handler.py` | Add `extract_paper_id()`, `deduplicate_evidence()`, integrate into `execute()` |
| `tests/unit/tools/test_search_handler.py` | Add comprehensive deduplication tests |
| `tests/unit/tools/test_openalex.py` | Add test for PMID extraction |

---

## Acceptance Criteria

- [ ] `openalex.py` extracts PMID from `work.ids.pmid` and stores in `Evidence.metadata`
- [ ] `extract_paper_id()` checks `Evidence.metadata.pmid` first (for OpenAlex cross-dedup)
- [ ] `extract_paper_id()` correctly parses all URL patterns:
  - PubMed URLs
  - Europe PMC MED, PMC, PPR, and PAT URLs
  - DOIs (normalized without trailing slash)
  - OpenAlex W-IDs (fallback)
  - ClinicalTrials NCT IDs (modern and legacy formats)
- [ ] `deduplicate_evidence()` removes duplicates while preserving source priority
- [ ] Unit tests cover all edge cases including OpenAlex with/without PMID
- [ ] Integration test confirms real searches are deduplicated
- [ ] Logging shows deduplication metrics

---

## Known Limitations

1. **OpenAlex without PMID**: Some OpenAlex papers (especially older or non-biomedical) may not have PMIDs. These will NOT be deduplicated against PubMed. This is acceptable - we prefer false negatives (keeping duplicates) over false positives (removing unique papers).

2. **PMC vs PMID**: PubMed Central IDs (PMC1234567) are different from PubMed IDs (12345678). A paper may have both, but we treat them as different identifiers. This may result in some duplicates not being caught when Europe PMC returns a PMC URL and PubMed returns a PMID URL for the same paper. Future enhancement: Use NCBI ID Converter API.

3. **Preprint → Peer-Reviewed**: When a preprint (PPR ID) is later published as a peer-reviewed paper (PMID), they will have different IDs and won't be deduplicated. This is intentional - users may want to see both versions.

---

## Rollback Plan

If deduplication causes issues:
1. Remove the `deduplicate_evidence()` call from `execute()`
2. All tests should still pass (deduplication is additive)
3. No need to revert OpenAlex PMID extraction (metadata only)

---

## References

- GitHub Issue #94
- [OpenAlex Work Object Documentation](https://docs.openalex.org/api-entities/works/work-object)
- [Europe PMC Help - ID Types](https://europepmc.org/help)
- [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api)
- `TOOL_ANALYSIS_CRITICAL.md` - "Cross-Tool Issues > Issue 1: MASSIVE DUPLICATION"
