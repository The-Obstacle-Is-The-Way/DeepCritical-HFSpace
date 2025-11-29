# DeepBoner Data Sources: Roadmap Summary

**Created**: 2024-11-27
**Purpose**: Future maintainability and hackathon continuation

---

## Current State

### Working Tools

| Tool | Status | Data Quality |
|------|--------|--------------|
| PubMed | ✅ Works | Good (abstracts only) |
| ClinicalTrials.gov | ✅ Works | Good (filtered for interventional) |
| Europe PMC | ✅ Works | Good (includes preprints) |

### Removed Tools

| Tool | Status | Reason |
|------|--------|--------|
| bioRxiv | ❌ Removed | No search API - only date/DOI lookup |

---

## Priority Improvements

### P0: Critical (Do First)

1. **Add Rate Limiting to PubMed**
   - NCBI will block us without it
   - Use `limits` library (see reference repo)
   - 3/sec without key, 10/sec with key

### P1: High Value, Medium Effort

2. **Add OpenAlex as 4th Source**
   - Citation network (huge for drug repurposing)
   - Concept tagging (semantic discovery)
   - Already implemented in reference repo
   - Free, no API key

3. **PubMed Full-Text via BioC**
   - Get full paper text for PMC papers
   - Already in reference repo

### P2: Nice to Have

4. **ClinicalTrials.gov Results**
   - Get efficacy data from completed trials
   - Requires more complex API calls

5. **Europe PMC Annotations**
   - Text-mined entities (genes, drugs, diseases)
   - Automatic entity extraction

---

## Effort Estimates

| Improvement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| PubMed rate limiting | 1 hour | Stability | P0 |
| OpenAlex basic search | 2 hours | High | P1 |
| OpenAlex citations | 2 hours | Very High | P1 |
| PubMed full-text | 3 hours | Medium | P1 |
| CT.gov results | 4 hours | Medium | P2 |
| Europe PMC annotations | 3 hours | Medium | P2 |

---

## Architecture Decision

### Option A: Keep Current + Add OpenAlex

```
                    User Query
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
 PubMed          ClinicalTrials        Europe PMC
 (abstracts)     (trials only)         (preprints)
    ↓                   ↓                   ↓
    └───────────────────┼───────────────────┘
                        ↓
                   OpenAlex              ← NEW
               (citations, concepts)
                        ↓
                  Orchestrator
                        ↓
                     Report
```

**Pros**: Low risk, additive
**Cons**: More complexity, some overlap

### Option B: OpenAlex as Primary

```
                    User Query
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
 OpenAlex          ClinicalTrials      Europe PMC
 (primary          (trials only)       (full-text
  search)                               fallback)
    ↓                   ↓                   ↓
    └───────────────────┼───────────────────┘
                        ↓
                  Orchestrator
                        ↓
                     Report
```

**Pros**: Simpler, citation network built-in
**Cons**: Lose some PubMed-specific features

### Recommendation: Option A

Keep current architecture working, add OpenAlex incrementally.

---

## Quick Wins (Can Do Today)

1. **Add `limits` to `pyproject.toml`**
   ```toml
   dependencies = [
       "limits>=3.0",
   ]
   ```

2. **Copy OpenAlex tool from reference repo**
   - File: `reference_repos/DeepBoner/DeepResearch/src/tools/openalex_tools.py`
   - Adapt to our `SearchTool` base class

3. **Enable NCBI API Key**
   - Add to `.env`: `NCBI_API_KEY=your_key`
   - 10x rate limit improvement

---

## External Resources Worth Exploring

### Python Libraries

| Library | For | Notes |
|---------|-----|-------|
| `limits` | Rate limiting | Used by reference repo |
| `pyalex` | OpenAlex wrapper | [GitHub](https://github.com/J535D165/pyalex) |
| `metapub` | PubMed | Full-featured |
| `sentence-transformers` | Semantic search | For embeddings |

### APIs Not Yet Used

| API | Provides | Effort |
|-----|----------|--------|
| RxNorm | Drug name normalization | Low |
| DrugBank | Drug targets/mechanisms | Medium (license) |
| UniProt | Protein data | Medium |
| ChEMBL | Bioactivity data | Medium |

### RAG Tools (Future)

| Tool | Purpose |
|------|---------|
| [PaperQA](https://github.com/Future-House/paper-qa) | RAG for scientific papers |
| [txtai](https://github.com/neuml/txtai) | Embeddings + search |
| [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | Biomedical embeddings |

---

## Files in This Directory

| File | Contents |
|------|----------|
| `00_ROADMAP_SUMMARY.md` | This file |
| `01_PUBMED_IMPROVEMENTS.md` | PubMed enhancement details |
| `02_CLINICALTRIALS_IMPROVEMENTS.md` | ClinicalTrials.gov details |
| `03_EUROPEPMC_IMPROVEMENTS.md` | Europe PMC details |
| `04_OPENALEX_INTEGRATION.md` | OpenAlex integration plan |

---

## For Future Maintainers

If you're picking this up after the hackathon:

1. **Start with OpenAlex** - biggest bang for buck
2. **Add rate limiting** - prevents API blocks
3. **Don't bother with bioRxiv** - use Europe PMC instead
4. **Reference repo is gold** - `reference_repos/DeepBoner/` has working implementations

Good luck! 🚀
