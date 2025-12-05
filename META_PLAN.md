# META_PLAN: DeepBoner Stabilization Roadmap

**Created**: 2025-12-03
**Updated**: 2025-12-04
**Status**: ✅ COMPLETE
**Purpose**: Single source of truth for what to do next before adding features

---

## Executive Summary

**Codebase Health**: PRODUCTION-READY
- 302 tests passing
- No type errors (mypy clean)
- No linting issues (ruff clean)
- All P3 tech debt resolved

**Key Finding**: Architecture is sound. All specs implemented. Tech debt cleaned.

**Status**: Steps 1-3 COMPLETE. Ready for feature development or documentation polish.

---

## Current State Assessment

### Documentation Status

| Document | Status | Action |
|----------|--------|--------|
| `docs/STATUS_LLAMAINDEX_INTEGRATION.md` | DONE | Keep as-is |
| `docs/specs/archive/SPEC_13_EVIDENCE_DEDUPLICATION.md` | ✅ IMPLEMENTED | Archived |
| `docs/specs/archive/SPEC_14_CLINICALTRIALS_OUTCOMES.md` | ✅ IMPLEMENTED | Archived |
| `docs/future-roadmap/TOOL_ANALYSIS_CRITICAL.md` | ANALYSIS DONE | Reference for future |
| `docs/ARCHITECTURE.md` | PARTIAL | Expand with diagrams |
| `docs/architecture/system_registry.md` | DONE | Canonical SSOT for wiring |

### Architecture Status

| Component | Status | Notes |
|-----------|--------|-------|
| `src/orchestrators/` | COMPLETE | Factory pattern, protocols |
| `src/clients/` | COMPLETE | OpenAI/HuggingFace working, Provider Registry pattern |
| `src/tools/` | COMPLETE | Deduplication + outcomes extraction done |
| `src/agents/` | FUNCTIONAL | All agents wired, some experimental |
| `src/services/` | COMPLETE | Embeddings, RAG, memory all working |

### Open Issues

| Issue | Priority | Effort |
|-------|----------|--------|
| ~~Evidence deduplication (SPEC_13)~~ | ~~HIGH~~ | ✅ DONE |
| ~~ClinicalTrials outcomes (SPEC_14)~~ | ~~HIGH~~ | ✅ DONE |
| ~~Remove Anthropic wiring (P3)~~ | ~~P3~~ | ✅ DONE (PR #130) |
| ~~Remove Modal wiring (P3)~~ | ~~P3~~ | ✅ DONE (PR #130) |
| Expand ARCHITECTURE.md | LOW | 2 hours |
| P3 Progress Bar positioning | P3 | 30 min |

---

## The Next 5 Steps

### ~~Step 1: Implement SPEC_13 - Evidence Deduplication~~ ✅ COMPLETE
**Priority**: ~~HIGH~~ DONE | **Effort**: ~~3-4 hours~~ | **Impact**: 30-50% token savings

✅ **COMPLETED** - Deduplication now removes duplicate papers from PubMed/Europe PMC/OpenAlex.

**Files modified**:
- `src/tools/search_handler.py` - Added `extract_paper_id()` and `deduplicate_evidence()`
- `src/tools/openalex.py` - Extracts PMID from `work.ids.pmid`
- `tests/unit/tools/test_search_handler.py` - 22 dedup tests
- `tests/integration/test_search_deduplication.py` - Integration test

**Spec**: `docs/specs/archive/SPEC_13_EVIDENCE_DEDUPLICATION.md` (Status: Implemented)

---

### ~~Step 2: Implement SPEC_14 - ClinicalTrials Outcomes~~ ✅ COMPLETE
**Priority**: ~~HIGH~~ DONE | **Effort**: ~~2-3 hours~~ | **Impact**: Critical efficacy data

✅ **COMPLETED** - ClinicalTrials now extracts outcome measures and results status.

**Files modified**:
- `src/tools/clinicaltrials.py` - Added `OutcomesModule`, `HasResults` fields, `_extract_primary_outcome()`
- `tests/unit/tools/test_clinicaltrials.py` - 4 outcome tests + 2 integration tests

**Spec**: `docs/specs/archive/SPEC_14_CLINICALTRIALS_OUTCOMES.md` (Status: Implemented)

---

### ~~Step 3: Remove Anthropic + Modal Tech Debt~~ ✅ COMPLETE
**Priority**: ~~P3~~ DONE | **Effort**: ~~1 hour~~ | **Impact**: Code clarity

✅ **COMPLETED** - Both Anthropic and Modal partial integrations removed in PR #130.

**Files removed/modified**:
- `src/utils/llm_factory.py` - DELETED
- `src/tools/code_execution.py` - DELETED
- `src/services/statistical_analyzer.py` - DELETED
- `src/agents/analysis_agent.py`, `code_executor_agent.py` - DELETED
- All config, factory, and agent files cleaned

**Docs archived**: `docs/bugs/archive/P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md`, `P3_MODAL_INTEGRATION_REMOVAL.md`

---

### Step 4: Documentation Consolidation
**Priority**: MEDIUM | **Effort**: 2 hours | **Impact**: Developer clarity

Create single canonical architecture doc with:
- System flow diagram
- Component interaction map
- Error handling patterns
- Deployment topology

**Output**: Expanded `docs/ARCHITECTURE.md`

---

### Step 5: Create Implementation Status Matrix
**Priority**: LOW | **Effort**: 1 hour | **Impact**: Project tracking

Update `docs/index.md` or create `docs/IMPLEMENTATION_STATUS.md` with:
- Phase completion tracking (14 phases)
- Post-hackathon roadmap status
- Clear DONE vs TODO markers

---

## What NOT To Do (Yet)

1. **Add new features** - Stabilize first
2. **Add new LLM providers** - OpenAI/HuggingFace cover all use cases
3. **Build Neo4j knowledge graph** - Overkill for current needs
4. **Implement full-text retrieval** - Phase 15+ (after stabilization)
5. **Add MeSH term expansion** - Phase 15+ (optimization)

---

## Documentation Sprawl Analysis

**Total docs**: 91 markdown files in `docs/`

**Organization**:
```text
docs/
├── architecture/      # Canonical architecture docs (4 files)
├── brainstorming/     # Ideas, not commitments (6 files)
├── bugs/              # Active bugs + archive (25+ files)
├── decisions/         # ADRs from Nov 2025 (2 files)
├── development/       # Dev guides (1 file)
├── future-roadmap/    # Deferred work (5 files)
├── guides/            # User guides (1 file)
├── implementation/    # Phase docs 1-14 (15 files)
├── specs/             # Feature specs (4 files)
├── ARCHITECTURE.md    # High-level overview
└── index.md           # Entry point
```

**Recommendation**: Structure is fine. Both SPEC_13 and SPEC_14 are now implemented.

---

## Success Criteria

After completing Steps 1-5:

- [x] Evidence deduplication reduces duplicate papers by 80%+ ✅
- [x] ClinicalTrials shows outcome measures and results status ✅
- [x] No Anthropic references in codebase ✅ (PR #130)
- [x] No Modal references in codebase ✅ (PR #130)
- [ ] ARCHITECTURE.md has flow diagrams (optional)
- [ ] All 14 implementation phases marked DONE/TODO (optional)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-03 | Implement specs before doc cleanup | Specs are ready, high impact |
| 2025-12-03 | Remove Anthropic over adding Gemini | Tech debt cleanup > new features |
| 2025-12-03 | Defer full-text retrieval | Stabilize core first |
| 2025-12-03 | Mark SPEC_13 complete | All acceptance criteria verified, PR #122 |
| 2025-12-03 | Mark SPEC_14 complete | All acceptance criteria verified (was already implemented) |

---

## References

- `docs/architecture/system_registry.md` - Decorator/marker/tool wiring SSOT
- `docs/bugs/ACTIVE_BUGS.md` - Current bug tracking
- `CLAUDE.md` - Development commands and patterns
