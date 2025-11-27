# Europe PMC Tool: Current State & Future Improvements

**Status**: Currently Implemented (Replaced bioRxiv)
**Priority**: High (Preprint + Open Access Source)

---

## Why Europe PMC Over bioRxiv?

### bioRxiv API Limitations (Why We Abandoned It)

1. **No Search API**: Only returns papers by date range or DOI
2. **No Query Capability**: Cannot search for "metformin cancer"
3. **Workaround Required**: Would need to download ALL preprints and build local search
4. **Known Issue**: [Gradio Issue #8861](https://github.com/gradio-app/gradio/issues/8861) documents the limitation

### Europe PMC Advantages

1. **Full Search API**: Boolean queries, filters, facets
2. **Aggregates bioRxiv**: Includes bioRxiv, medRxiv content anyway
3. **Includes PubMed**: Also has MEDLINE content
4. **34 Preprint Servers**: Not just bioRxiv
5. **Open Access Focus**: Full-text when available

---

## Current Implementation

### What We Have (`src/tools/europepmc.py`)

- REST API search via `europepmc.org/webservices/rest/search`
- Preprint flagging via `firstPublicationDate` heuristics
- Returns: title, abstract, authors, DOI, source
- Marks preprints for transparency

### Current Limitations

1. **No Full-Text Retrieval**: Only metadata/abstracts
2. **No Citation Network**: Missing references/citations
3. **No Supplementary Files**: Not fetching figures/data
4. **Basic Preprint Detection**: Heuristic, not explicit flag

---

## Europe PMC API Capabilities

### Endpoints We Could Use

| Endpoint | Purpose | Currently Using |
|----------|---------|-----------------|
| `/search` | Query papers | Yes |
| `/fulltext/{ID}` | Full text (XML/JSON) | No |
| `/{PMCID}/supplementaryFiles` | Figures, data | No |
| `/citations/{ID}` | Who cited this | No |
| `/references/{ID}` | What this cites | No |
| `/annotations` | Text-mined entities | No |

### Rich Query Syntax

```python
# Current simple query
query = "metformin cancer"

# Could use advanced syntax
query = "(TITLE:metformin OR ABSTRACT:metformin) AND (cancer OR oncology)"
query += " AND (SRC:PPR)"  # Only preprints
query += " AND (FIRST_PDATE:[2023-01-01 TO 2024-12-31])"  # Date range
query += " AND (OPEN_ACCESS:y)"  # Only open access
```

### Source Filters

```python
# Filter by source
"SRC:MED"     # MEDLINE
"SRC:PMC"     # PubMed Central
"SRC:PPR"     # Preprints (bioRxiv, medRxiv, etc.)
"SRC:AGR"     # Agricola
"SRC:CBA"     # Chinese Biological Abstracts
```

---

## Recommended Improvements

### Phase 1: Rich Metadata

```python
# Add to search results
additional_fields = [
    "citedByCount",           # Impact indicator
    "source",                 # Explicit source (MED, PMC, PPR)
    "isOpenAccess",           # Boolean flag
    "fullTextUrlList",        # URLs for full text
    "authorAffiliations",     # Institution info
    "grantsList",             # Funding info
]
```

### Phase 2: Full-Text Retrieval

```python
async def get_fulltext(pmcid: str) -> str | None:
    """Get full text for open access papers."""
    # XML format
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    # Or JSON
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextJSON"
```

### Phase 3: Citation Network

```python
async def get_citations(pmcid: str) -> list[str]:
    """Get papers that cite this one."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/citations"

async def get_references(pmcid: str) -> list[str]:
    """Get papers this one cites."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/references"
```

### Phase 4: Text-Mined Annotations

Europe PMC extracts entities automatically:

```python
async def get_annotations(pmcid: str) -> dict:
    """Get text-mined entities (genes, diseases, drugs)."""
    url = f"https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
    params = {
        "articleIds": f"PMC:{pmcid}",
        "type": "Gene_Proteins,Diseases,Chemicals",
        "format": "JSON",
    }
    # Returns structured entity mentions with positions
```

---

## Supplementary File Retrieval

From reference repo (`bioinformatics_tools.py` lines 123-149):

```python
def get_figures(pmcid: str) -> dict[str, str]:
    """Download figures and supplementary files."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles?includeInlineImage=true"
    # Returns ZIP with images, returns base64-encoded
```

---

## Preprint-Specific Features

### Identify Preprint Servers

```python
PREPRINT_SOURCES = {
    "PPR": "General preprints",
    "bioRxiv": "Biology preprints",
    "medRxiv": "Medical preprints",
    "chemRxiv": "Chemistry preprints",
    "Research Square": "Multi-disciplinary",
    "Preprints.org": "MDPI preprints",
}

# Check if published version exists
async def check_published_version(preprint_doi: str) -> str | None:
    """Check if preprint has been peer-reviewed and published."""
    # Europe PMC links preprints to final versions
```

---

## Rate Limiting

Europe PMC is more generous than NCBI:

```python
# No documented hard limit, but be respectful
# Recommend: 10-20 requests/second max
# Use email in User-Agent for polite pool
headers = {
    "User-Agent": "DeepCritical/1.0 (mailto:your@email.com)"
}
```

---

## vs. The Lens & OpenAlex

| Feature | Europe PMC | The Lens | OpenAlex |
|---------|------------|----------|----------|
| Biomedical Focus | Yes | Partial | Partial |
| Preprints | Yes (34 servers) | Yes | Yes |
| Full Text | PMC papers | Links | No |
| Citations | Yes | Yes | Yes |
| Annotations | Yes (text-mined) | No | No |
| Rate Limits | Generous | Moderate | Very generous |
| API Key | Optional | Required | Optional |

---

## Sources

- [Europe PMC REST API](https://europepmc.org/RestfulWebService)
- [Europe PMC Annotations API](https://europepmc.org/AnnotationsApi)
- [Europe PMC Articles API](https://europepmc.org/ArticlesApi)
- [rOpenSci medrxivr](https://docs.ropensci.org/medrxivr/)
- [bioRxiv TDM Resources](https://www.biorxiv.org/tdm)
