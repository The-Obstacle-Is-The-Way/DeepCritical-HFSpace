# Implementation Specs

> Last updated: 2025-12-03
>
> **Note:** Implemented specs archived to `docs/specs/archive/`

---

## Active Specs (Not Yet Implemented)

### SPEC_13: Evidence Deduplication

**File:** `SPEC_13_EVIDENCE_DEDUPLICATION.md`
**Status:** Draft - Ready for Implementation
**Priority:** P1
**GitHub Issue:** #94

**Problem:** Same paper appears 2-3 times because Europe PMC and OpenAlex both index PubMed.

**Solution:** Extract PMID from OpenAlex metadata, deduplicate by paper ID in SearchHandler.

---

### SPEC_14: ClinicalTrials Outcomes

**File:** `SPEC_14_CLINICALTRIALS_OUTCOMES.md`
**Status:** Draft - Ready for Implementation
**Priority:** P1
**GitHub Issue:** #95

**Problem:** ClinicalTrials tool misses critical efficacy data (primary outcomes, results availability).

**Solution:** Add `OutcomesModule` and `HasResults` to API fields, extract outcome measures.

---

## Implemented Specs (Archived)

All implemented specs are in `docs/specs/archive/`. Summary:

| Spec | Description | PR/Commit |
|------|-------------|-----------|
| SPEC_01 | Demo Termination | Archived |
| SPEC_02 | E2E Testing | Archived |
| SPEC_03 | OpenAlex Integration | Archived |
| SPEC_04 | Magentic UX | Archived |
| SPEC_05 | Orchestrator Cleanup | Archived |
| SPEC_06 | Simple Mode Synthesis | Archived (deleted simple.py) |
| SPEC_07 | LangGraph Memory Arch | Archived |
| SPEC_08 | Integrate Memory Layer | Archived |
| SPEC_09 | LlamaIndex Integration | Archived |
| SPEC_10 | Domain Agnostic Refactor | Archived |
| SPEC_11 | Sexual Health Focus | Archived |
| SPEC_12 | Narrative Synthesis | Archived |
| **SPEC_15** | Advanced Mode Performance | PR #65 - max_rounds=5, early termination |
| **SPEC_16** | Unified Chat Client Architecture | PR #115 - HuggingFace + OpenAI factory |
| **SPEC_17** | Accumulator Pattern | PR #117 - Fixes repr bug |

---

## How to Write a Spec

1. Create `docs/specs/SPEC_{N}_{NAME}.md`
2. Include: Problem, Solution, Test Plan, Acceptance Criteria
3. Link to GitHub issue
4. Update this README
5. When implemented, move to `archive/`
