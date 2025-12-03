# Critical Analysis: Search Tools - Limitations, Gaps, and Improvements

**Date**: November 2025
**Purpose**: Honest assessment of all search tools to identify what's working, what's broken, and what needs improvement WITHOUT horizontal sprawl.

---

## Executive Summary

DeepBoner currently has **4 search tools**:
1. PubMed (NCBI E-utilities)
2. ClinicalTrials.gov (API v2)
3. Europe PMC (includes preprints)
4. OpenAlex (citation-aware)

**Overall Assessment**: Tools are functional but have significant gaps in:
- Deduplication (PubMed ∩ Europe PMC ∩ OpenAlex = massive overlap)
- Full-text retrieval (only abstracts currently)
- Citation graph traversal (OpenAlex has data but we don't use it)
- Query optimization (basic synonym expansion, no MeSH term mapping)

---

## Tool 1: PubMed (NCBI E-utilities)

**File**: `src/tools/pubmed.py`

### What It Does Well
| Feature | Status | Notes |
|---------|--------|-------|
| Rate limiting | ✅ | Shared limiter, respects 3/sec (no key) or 10/sec (with key) |
| Retry logic | ✅ | tenacity with exponential backoff |
| Query preprocessing | ✅ | Strips question words, expands synonyms |
| Abstract parsing | ✅ | Handles XML edge cases (dict vs list) |

### Limitations (API-Level)
| Limitation | Severity | Workaround Possible? |
|------------|----------|---------------------|
| **10,000 result cap per query** | Medium | Yes - use date ranges to paginate |
| **Abstracts only** (no full text) | High | No - full text requires PMC or publisher |
| **No citation counts** | Medium | Yes - cross-reference with OpenAlex |
| **Rate limit (10/sec max)** | Low | Already handled |

### Current Implementation Gaps
```python
# GAP 1: No MeSH term expansion
# Current: expand_synonyms() uses hardcoded dict
# Better: Use NCBI's E-utilities to get MeSH terms for query

# GAP 2: No date filtering
# Current: Gets whatever PubMed returns (biased toward recent)
# Better: Add date range parameter for historical research

# GAP 3: No publication type filtering
# Current: Returns all types (reviews, case reports, RCTs)
# Better: Filter for RCTs and systematic reviews when appropriate
```

### Priority Improvements
1. **HIGH**: Add publication type filter (Reviews, RCTs, Meta-analyses)
2. **MEDIUM**: Add date range parameter
3. **LOW**: MeSH term expansion via E-utilities

---

## Tool 2: ClinicalTrials.gov

**File**: `src/tools/clinicaltrials.py`

### What It Does Well
| Feature | Status | Notes |
|---------|--------|-------|
| API v2 usage | ✅ | Modern API, not deprecated v1 |
| Interventional filter | ✅ | Only gets drug/treatment studies |
| Status filter | ✅ | COMPLETED, ACTIVE, RECRUITING |
| httpx → requests workaround | ✅ | Bypasses WAF TLS fingerprint block |

### Limitations (API-Level)
| Limitation | Severity | Workaround Possible? |
|------------|----------|---------------------|
| **No results data** | High | Yes - available via different endpoint |
| **No outcome measures** | High | Yes - add to FIELDS list |
| **No adverse events** | Medium | Yes - separate API call |
| **Sparse drug mechanism data** | Medium | No - not in API |

### Current Implementation Gaps
```python
# GAP 1: Missing critical fields
FIELDS: ClassVar[list[str]] = [
    "NCTId",
    "BriefTitle",
    "Phase",
    "OverallStatus",
    "Condition",
    "InterventionName",
    "StartDate",
    "BriefSummary",
    # MISSING:
    # "PrimaryOutcome",
    # "SecondaryOutcome",
    # "ResultsFirstSubmitDate",
    # "StudyResults",  # Whether results are posted
]

# GAP 2: No results retrieval
# Many completed trials have posted results
# We could get actual efficacy data, not just trial existence

# GAP 3: No linked publications
# Trials often link to PubMed articles with results
# We could follow these links for richer evidence
```

### Priority Improvements
1. **HIGH**: Add outcome measures to FIELDS
2. **HIGH**: Check for and retrieve posted results
3. **MEDIUM**: Follow linked publications (NCT → PMID)

---

## Tool 3: Europe PMC

**File**: `src/tools/europepmc.py`

### What It Does Well
| Feature | Status | Notes |
|---------|--------|-------|
| Preprint coverage | ✅ | bioRxiv, medRxiv, ChemRxiv indexed |
| Preprint labeling | ✅ | `[PREPRINT - Not peer-reviewed]` marker |
| DOI/PMID fallback URLs | ✅ | Smart URL construction |
| Relevance scoring | ✅ | Preprints weighted lower (0.75 vs 0.9) |

### Limitations (API-Level)
| Limitation | Severity | Workaround Possible? |
|------------|----------|---------------------|
| **No full text for most articles** | High | Partial - CC-licensed available after 14 days |
| **Citation data limited** | Medium | Only journal articles, not preprints |
| **Preprint-publication linking gaps** | Medium | ~50% of links missing per Crossref |
| **License info sometimes missing** | Low | Manual review required |

### Current Implementation Gaps
```python
# GAP 1: No full-text retrieval
# Europe PMC has full text for many CC-licensed articles
# Could retrieve full text XML via separate endpoint

# GAP 2: Massive overlap with PubMed
# Europe PMC indexes all of PubMed/MEDLINE
# We're getting duplicates with no deduplication

# GAP 3: No citation network
# Europe PMC has "citedByCount" but we don't use it
# Could prioritize highly-cited preprints
```

### Priority Improvements
1. **HIGH**: Add deduplication with PubMed (by PMID)
2. **MEDIUM**: Retrieve citation counts for ranking
3. **LOW**: Full-text retrieval for CC-licensed articles

---

## Tool 4: OpenAlex

**File**: `src/tools/openalex.py`

### What It Does Well
| Feature | Status | Notes |
|---------|--------|-------|
| Citation counts | ✅ | Sorted by `cited_by_count:desc` |
| Abstract reconstruction | ✅ | Handles inverted index format |
| Concept extraction | ✅ | Hierarchical classification |
| Open access detection | ✅ | `is_oa` and `pdf_url` |
| Polite pool | ✅ | mailto for 100k/day limit |
| Rich metadata | ✅ | Best metadata of all tools |

### Limitations (API-Level)
| Limitation | Severity | Workaround Possible? |
|------------|----------|---------------------|
| **Author truncation at 100** | Low | Only affects mega-author papers |
| **No full text** | High | No - OpenAlex is metadata only |
| **Stale data (1-2 day lag)** | Low | Acceptable for research |

### Current Implementation Gaps
```python
# GAP 1: No citation graph traversal
# OpenAlex has `cited_by` and `references` endpoints
# We could find seminal papers by following citation chains

# GAP 2: No related works
# OpenAlex has ML-powered "related_works" field
# Could expand search to similar papers

# GAP 3: No concept filtering
# OpenAlex has hierarchical concepts
# Could filter for specific domains (e.g., "Sexual health" concept)

# GAP 4: Overlap with PubMed
# OpenAlex indexes most of PubMed
# More duplicates without deduplication
```

### Priority Improvements
1. **HIGH**: Add citation graph traversal (find seminal papers)
2. **HIGH**: Add deduplication with PubMed/Europe PMC
3. **MEDIUM**: Use `related_works` for query expansion
4. **LOW**: Concept-based filtering

---

## Cross-Tool Issues

### Issue 1: MASSIVE DUPLICATION

```
PubMed: 36M+ articles
Europe PMC: Indexes ALL of PubMed + preprints
OpenAlex: 250M+ works (includes PubMed)

Current behavior: All 3 return the same papers
Result: Duplicate evidence, wasted tokens, inflated counts
```

**Solution**: Deduplication by PMID/DOI
```python
# Proposed: Add to SearchHandler
def deduplicate_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
    seen_ids: set[str] = set()
    unique: list[Evidence] = []
    for e in evidence_list:
        # Extract PMID or DOI from URL
        paper_id = extract_paper_id(e.citation.url)
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique.append(e)
    return unique
```

### Issue 2: NO FULL-TEXT RETRIEVAL

All tools return **abstracts only**. For deep research, this is limiting.

**What's Actually Possible**:
| Source | Full Text Access | How |
|--------|------------------|-----|
| PubMed Central (PMC) | Yes, for OA articles | Separate API: `efetch` with `db=pmc` |
| Europe PMC | Yes, CC-licensed after 14 days | `/fullTextXML/{id}` endpoint |
| OpenAlex | No | Metadata only |
| Unpaywall | Yes, OA link discovery | Separate API |

**Recommendation**: Add PMC full-text retrieval for open access articles.

### Issue 3: NO CITATION GRAPH

OpenAlex has rich citation data but we only use `cited_by_count` for sorting.

**Untapped Capabilities**:
- `cited_by`: Find papers that cite a key paper
- `references`: Find sources a paper cites
- `related_works`: ML-powered similar papers

**Use Case**: User asks about "testosterone therapy for HSDD". We find a seminal 2019 RCT. We could automatically find:
- Papers that cite it (newer evidence)
- Papers it cites (foundational research)
- Related papers (similar topics)

---

## What's NOT Possible (API Constraints)

| Feature | Why Not Possible |
|---------|------------------|
| **bioRxiv direct search** | No keyword search API, only RSS feed of latest |
| **arXiv search** | API exists but irrelevant for sexual health |
| **PubMed full text** | Requires publisher access or PMC |
| **Real-time trial results** | ClinicalTrials.gov results are static snapshots |
| **Drug mechanism data** | Not in any API - would need ChEMBL or DrugBank |

---

## Recommended Improvements (Priority Order)

### Phase 1: Fix Fundamentals (High ROI)
1. **Deduplication** - Stop returning the same paper 3 times
2. **Outcome measures in ClinicalTrials** - Get actual efficacy data
3. **Citation counts from all sources** - Rank by influence, not recency

### Phase 2: Depth Improvements (Medium ROI)
4. **PMC full-text retrieval** - Get full papers for OA articles
5. **Citation graph traversal** - Find seminal papers automatically
6. **Publication type filtering** - Prioritize RCTs and meta-analyses

### Phase 3: Quality Improvements (Lower ROI, Nice-to-Have)
7. **MeSH term expansion** - Better PubMed queries
8. **Related works expansion** - Use OpenAlex ML similarity
9. **Date range filtering** - Historical vs recent research

---

## Neo4j Integration (Future Consideration)

**Question**: Should we add Neo4j for citation graph storage?

**Answer**: Not yet. Here's why:

| Approach | Complexity | Value |
|----------|------------|-------|
| OpenAlex API for citation traversal | Low | High |
| Neo4j for local citation graph | High | Medium (unless doing graph analytics) |
| Cron job to sync OpenAlex → Neo4j | Medium | Only if we need offline access |

**Recommendation**: Use OpenAlex API for citation traversal first. Only add Neo4j if:
1. We need to do complex graph queries (PageRank on citations, community detection)
2. We need offline access to citation data
3. We're hitting OpenAlex rate limits

---

## Summary: What's Broken vs What's Working

### Working Well
- Basic search across all 4 sources
- Rate limiting and retry logic
- Query preprocessing
- Evidence model with citations

### Needs Fixing (Current Scope)
- Deduplication (critical)
- Outcome measures in ClinicalTrials (critical)
- Citation-based ranking (important)

### Future Enhancements (Out of Current Scope)
- Full-text retrieval
- Citation graph traversal
- Neo4j integration
- Drug mechanism data (would need new data sources)

---

## Sources

- [NCBI E-utilities Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [NCBI Rate Limits](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
- [OpenAlex API Docs](https://docs.openalex.org/)
- [OpenAlex Limitations](https://docs.openalex.org/api-entities/authors/limitations)
- [Europe PMC RESTful API](https://europepmc.org/RestfulWebService)
- [Europe PMC Preprints](https://pmc.ncbi.nlm.nih.gov/articles/PMC11426508/)
- [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api)
