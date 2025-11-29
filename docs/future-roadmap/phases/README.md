# Implementation Plans

TDD implementation plans based on the brainstorming documents. Each phase is a self-contained vertical slice with tests, implementation, and demo scripts.

---

## Prerequisites (COMPLETED)

The following foundational changes have been implemented to support all three phases:

| Change | File | Status |
|--------|------|--------|
| Add `"openalex"` to `SourceName` | `src/utils/models.py:9` | ✅ Done |
| Add `metadata` field to `Evidence` | `src/utils/models.py:39-42` | ✅ Done |
| Export all tools from `__init__.py` | `src/tools/__init__.py` | ✅ Done |

All 110 tests pass after these changes.

---

## Priority Order

| Phase | Name | Priority | Effort | Value |
|-------|------|----------|--------|-------|
| **17** | Rate Limiting | P0 CRITICAL | 1 hour | Stability |
| **15** | OpenAlex | HIGH | 2-3 hours | Very High |
| **16** | PubMed Full-Text | MEDIUM | 3 hours | High |

**Recommended implementation order**: 17 → 15 → 16

---

## Phase 15: OpenAlex Integration

**File**: [15_PHASE_OPENALEX.md](./15_PHASE_OPENALEX.md)

Add OpenAlex as 4th data source for:
- Citation networks (who cites whom)
- Concept tagging (semantic discovery)
- 209M+ scholarly works
- Free, no API key required

**Quick Start**:
```bash
# Create the tool
touch src/tools/openalex.py
touch tests/unit/tools/test_openalex.py

# Run tests first (TDD)
uv run pytest tests/unit/tools/test_openalex.py -v

# Demo
uv run python examples/openalex_demo.py
```

---

## Phase 16: PubMed Full-Text

**File**: [16_PHASE_PUBMED_FULLTEXT.md](./16_PHASE_PUBMED_FULLTEXT.md)

Add full-text retrieval via BioC API for:
- Complete paper text (not just abstracts)
- Structured sections (intro, methods, results)
- Better evidence for LLM synthesis

**Quick Start**:
```bash
# Add methods to existing pubmed.py
# Tests in test_pubmed_fulltext.py

# Run tests
uv run pytest tests/unit/tools/test_pubmed_fulltext.py -v

# Demo
uv run python examples/pubmed_fulltext_demo.py
```

---

## Phase 17: Rate Limiting

**File**: [17_PHASE_RATE_LIMITING.md](./17_PHASE_RATE_LIMITING.md)

Replace naive sleep-based rate limiting with `limits` library for:
- Moving window algorithm
- Shared limits across instances
- Configurable per-API rates
- Production-grade stability

**Quick Start**:
```bash
# Add dependency
uv add limits

# Create module
touch src/tools/rate_limiter.py
touch tests/unit/tools/test_rate_limiting.py

# Run tests
uv run pytest tests/unit/tools/test_rate_limiting.py -v

# Demo
uv run python examples/rate_limiting_demo.py
```

---

## TDD Workflow

Each implementation doc follows this pattern:

1. **Write tests first** - Define expected behavior
2. **Run tests** - Verify they fail (red)
3. **Implement** - Write minimal code to pass
4. **Run tests** - Verify they pass (green)
5. **Refactor** - Clean up if needed
6. **Demo** - Verify end-to-end with real APIs
7. **`make check`** - Ensure no regressions

---

## Related Brainstorming Docs

These implementation plans are derived from:

- [00_ROADMAP_SUMMARY.md](../00_ROADMAP_SUMMARY.md) - Priority overview
- [01_PUBMED_IMPROVEMENTS.md](../01_PUBMED_IMPROVEMENTS.md) - PubMed details
- [02_CLINICALTRIALS_IMPROVEMENTS.md](../02_CLINICALTRIALS_IMPROVEMENTS.md) - CT.gov details
- [03_EUROPEPMC_IMPROVEMENTS.md](../03_EUROPEPMC_IMPROVEMENTS.md) - Europe PMC details
- [04_OPENALEX_INTEGRATION.md](../04_OPENALEX_INTEGRATION.md) - OpenAlex integration

---

## Future Phases (Not Yet Documented)

Based on brainstorming, these could be added later:

- **Phase 18**: ClinicalTrials.gov Results Retrieval
- **Phase 19**: Europe PMC Annotations API
- **Phase 20**: Drug Name Normalization (RxNorm)
- **Phase 21**: Citation Network Queries (OpenAlex)
- **Phase 22**: Semantic Search with Embeddings
