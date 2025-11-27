# PubMed Tool: Current State & Future Improvements

**Status**: Currently Implemented
**Priority**: High (Core Data Source)

---

## Current Implementation

### What We Have (`src/tools/pubmed.py`)

- Basic E-utilities search via `esearch.fcgi` and `efetch.fcgi`
- Query preprocessing (strips question words, expands synonyms)
- Returns: title, abstract, authors, journal, PMID
- Rate limiting: None implemented (relying on NCBI defaults)

### Current Limitations

1. **No Full-Text Access**: Only retrieves abstracts, not full paper text
2. **No Rate Limiting**: Risk of being blocked by NCBI
3. **No BioC Format**: Missing structured full-text extraction
4. **No Figure Retrieval**: No supplementary materials access
5. **No PMC Integration**: Missing open-access full-text via PMC

---

## Reference Implementation (DeepCritical Reference Repo)

The reference repo at `reference_repos/DeepCritical/DeepResearch/src/tools/bioinformatics_tools.py` has a more sophisticated implementation:

### Features We're Missing

```python
# Rate limiting (lines 47-50)
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

storage = MemoryStorage()
limiter = MovingWindowRateLimiter(storage)
rate_limit = parse("3/second")  # NCBI allows 3/sec without API key, 10/sec with

# Full-text via BioC format (lines 108-120)
def _get_fulltext(pmid: int) -> dict[str, Any] | None:
    pmid_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"
    # Returns structured JSON with full text for open-access papers

# Figure retrieval via Europe PMC (lines 123-149)
def _get_figures(pmcid: str) -> dict[str, str]:
    suppl_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
    # Returns base64-encoded images from supplementary materials
```

---

## Recommended Improvements

### Phase 1: Rate Limiting (Critical)

```python
# Add to src/tools/pubmed.py
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

storage = MemoryStorage()
limiter = MovingWindowRateLimiter(storage)

# With NCBI_API_KEY: 10/sec, without: 3/sec
def get_rate_limit():
    if settings.ncbi_api_key:
        return parse("10/second")
    return parse("3/second")
```

**Dependencies**: `pip install limits`

### Phase 2: Full-Text Retrieval

```python
async def get_fulltext(pmid: str) -> str | None:
    """Get full text for open-access papers via BioC API."""
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"
    # Only works for PMC papers (open access)
```

### Phase 3: PMC ID Resolution

```python
async def get_pmc_id(pmid: str) -> str | None:
    """Convert PMID to PMCID for full-text access."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
```

---

## Python Libraries to Consider

| Library | Purpose | Notes |
|---------|---------|-------|
| [Biopython](https://biopython.org/) | `Bio.Entrez` module | Official, well-maintained |
| [PyMed](https://pypi.org/project/pymed/) | PubMed wrapper | Simpler API, less control |
| [metapub](https://pypi.org/project/metapub/) | Full-featured | Tested on 1/3 of PubMed |
| [limits](https://pypi.org/project/limits/) | Rate limiting | Used by reference repo |

---

## API Endpoints Reference

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `esearch.fcgi` | Search for PMIDs | 3/sec (10 with key) |
| `efetch.fcgi` | Fetch metadata | 3/sec (10 with key) |
| `esummary.fcgi` | Quick metadata | 3/sec (10 with key) |
| `pmcoa.cgi/BioC_json` | Full text (PMC only) | Unknown |
| `idconv/v1.0` | PMID ↔ PMCID | Unknown |

---

## Sources

- [PubMed E-utilities Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [NCBI BioC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/)
- [Searching PubMed with Python](https://marcobonzanini.com/2015/01/12/searching-pubmed-with-python/)
- [PyMed on PyPI](https://pypi.org/project/pymed/)
