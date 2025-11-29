# OpenAlex Integration: The Missing Piece?

**Status**: NOT Implemented (Candidate for Addition)
**Priority**: HIGH - Could Replace Multiple Tools
**Reference**: Already implemented in `reference_repos/DeepBoner`

---

## What is OpenAlex?

OpenAlex is a **fully open** index of the global research system:

- **209M+ works** (papers, books, datasets)
- **2B+ author records** (disambiguated)
- **124K+ venues** (journals, repositories)
- **109K+ institutions**
- **65K+ concepts** (hierarchical, linked to Wikidata)

**Free. Open. No API key required.**

---

## Why OpenAlex for DeepBoner?

### Current Architecture

```
User Query
    ↓
┌──────────────────────────────────────┐
│  PubMed    ClinicalTrials  Europe PMC │  ← 3 separate APIs
└──────────────────────────────────────┘
    ↓
Orchestrator (deduplicate, judge, synthesize)
```

### With OpenAlex

```
User Query
    ↓
┌──────────────────────────────────────┐
│              OpenAlex                 │  ← Single API
│  (includes PubMed + preprints +       │
│   citations + concepts + authors)     │
└──────────────────────────────────────┘
    ↓
Orchestrator (enrich with CT.gov for trials)
```

**OpenAlex already aggregates**:
- PubMed/MEDLINE
- Crossref
- ORCID
- Unpaywall (open access links)
- Microsoft Academic Graph (legacy)
- Preprint servers

---

## Reference Implementation

From `reference_repos/DeepBoner/DeepResearch/src/tools/openalex_tools.py`:

```python
class OpenAlexFetchTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="openalex_fetch",
                description="Fetch OpenAlex work or author",
                inputs={"entity": "TEXT", "identifier": "TEXT"},
                outputs={"result": "JSON"},
            )
        )

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        entity = params["entity"]      # "works", "authors", "venues"
        identifier = params["identifier"]
        base = "https://api.openalex.org"
        url = f"{base}/{entity}/{identifier}"
        resp = requests.get(url, timeout=30)
        return ExecutionResult(success=True, data={"result": resp.json()})
```

---

## OpenAlex API Features

### Search Works (Papers)

```python
# Search for metformin + cancer papers
url = "https://api.openalex.org/works"
params = {
    "search": "metformin cancer drug repurposing",
    "filter": "publication_year:>2020,type:article",
    "sort": "cited_by_count:desc",
    "per_page": 50,
}
```

### Rich Filtering

```python
# Filter examples
"publication_year:2023"
"type:article"                      # vs preprint, book, etc.
"is_oa:true"                        # Open access only
"concepts.id:C71924100"             # Papers about "Medicine"
"authorships.institutions.id:I27837315"  # From Harvard
"cited_by_count:>100"               # Highly cited
"has_fulltext:true"                 # Full text available
```

### What You Get Back

```json
{
    "id": "W2741809807",
    "title": "Metformin: A candidate drug for...",
    "publication_year": 2023,
    "type": "article",
    "cited_by_count": 45,
    "is_oa": true,
    "primary_location": {
        "source": {"display_name": "Nature Medicine"},
        "pdf_url": "https://...",
        "landing_page_url": "https://..."
    },
    "concepts": [
        {"id": "C71924100", "display_name": "Medicine", "score": 0.95},
        {"id": "C54355233", "display_name": "Pharmacology", "score": 0.88}
    ],
    "authorships": [
        {
            "author": {"id": "A123", "display_name": "John Smith"},
            "institutions": [{"display_name": "Harvard Medical School"}]
        }
    ],
    "referenced_works": ["W123", "W456"],  # Citations
    "related_works": ["W789", "W012"]       # Similar papers
}
```

---

## Key Advantages Over Current Tools

### 1. Citation Network (We Don't Have This!)

```python
# Get papers that cite a work
url = f"https://api.openalex.org/works?filter=cites:{work_id}"

# Get papers cited by a work
# Already in `referenced_works` field
```

### 2. Concept Tagging (We Don't Have This!)

OpenAlex auto-tags papers with hierarchical concepts:
- "Medicine" → "Pharmacology" → "Drug Repurposing"
- Can search by concept, not just keywords

### 3. Author Disambiguation (We Don't Have This!)

```python
# Find all works by an author
url = f"https://api.openalex.org/works?filter=authorships.author.id:{author_id}"
```

### 4. Institution Tracking

```python
# Find drug repurposing papers from top institutions
url = "https://api.openalex.org/works"
params = {
    "search": "drug repurposing",
    "filter": "authorships.institutions.id:I27837315",  # Harvard
}
```

### 5. Related Works

Each paper comes with `related_works` - semantically similar papers discovered by OpenAlex's ML.

---

## Proposed Implementation

### New Tool: `src/tools/openalex.py`

```python
"""OpenAlex search tool for comprehensive scholarly data."""

import httpx
from src.tools.base import SearchTool
from src.utils.models import Evidence

class OpenAlexTool(SearchTool):
    """Search OpenAlex for scholarly works with rich metadata."""

    name = "openalex"

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "filter": "type:article,is_oa:true",
                    "sort": "cited_by_count:desc",
                    "per_page": max_results,
                    "mailto": "deepboner@example.com",  # Polite pool
                },
            )
            data = resp.json()

        return [
            Evidence(
                source="openalex",
                title=work["title"],
                abstract=work.get("abstract", ""),
                url=work["primary_location"]["landing_page_url"],
                metadata={
                    "cited_by_count": work["cited_by_count"],
                    "concepts": [c["display_name"] for c in work["concepts"][:5]],
                    "is_open_access": work["is_oa"],
                    "pdf_url": work["primary_location"].get("pdf_url"),
                },
            )
            for work in data["results"]
        ]
```

---

## Rate Limits

OpenAlex is **extremely generous**:

- No hard rate limit documented
- Recommended: <100,000 requests/day
- **Polite pool**: Add `mailto=your@email.com` param for faster responses
- No API key required (optional for priority support)

---

## Should We Add OpenAlex?

### Arguments FOR

1. **Already in reference repo** - proven pattern
2. **Richer data** - citations, concepts, authors
3. **Single source** - reduces API complexity
4. **Free & open** - no keys, no limits
5. **Institution adoption** - Leiden, Sorbonne switched to it

### Arguments AGAINST

1. **Adds complexity** - another data source
2. **Overlap** - duplicates some PubMed data
3. **Not biomedical-focused** - covers all disciplines
4. **No full text** - still need PMC/Europe PMC for that

### Recommendation

**Add OpenAlex as a 4th source**, don't replace existing tools.

Use it for:
- Citation network analysis
- Concept-based discovery
- High-impact paper finding
- Author/institution tracking

Keep PubMed, ClinicalTrials, Europe PMC for:
- Authoritative biomedical search
- Clinical trial data
- Full-text access
- Preprint tracking

---

## Implementation Priority

| Task | Effort | Value |
|------|--------|-------|
| Basic search | Low | High |
| Citation network | Medium | Very High |
| Concept filtering | Low | High |
| Related works | Low | High |
| Author tracking | Medium | Medium |

---

## Sources

- [OpenAlex Documentation](https://docs.openalex.org)
- [OpenAlex API Overview](https://docs.openalex.org/api)
- [OpenAlex Wikipedia](https://en.wikipedia.org/wiki/OpenAlex)
- [Leiden University Announcement](https://www.leidenranking.com/information/openalex)
- [OpenAlex: A fully-open index (Paper)](https://arxiv.org/abs/2205.01833)
