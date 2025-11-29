# P0 Critical Bugs - DeepBoner Demo Broken

**Date**: 2025-11-28
**Status**: RESOLVED (2025-11-29)
**Priority**: P0 - Blocking hackathon submission

---

## Summary

The Gradio demo was non-functional due to 4 critical bugs. All have been fixed and verified.

---

## Bug 1: Free Tier LLM Quota Exhausted (P0) - FIXED

**Resolution**:
- Implemented `QuotaExhaustedError` detection in `HFInferenceJudgeHandler`.
- The agent now gracefully stops and displays a clear "Free Tier Quota Exceeded" message instead of looping infinitely.

## Bug 2: Evidence Counter Shows 0 After Dedup (P1) - FIXED

**Resolution**:
- Fixed by resolving Bug 4 (Data Leak). Deduplication now works correctly on isolated per-request collections.

## Bug 3: API Key Not Passed to Advanced Mode (P0) - FIXED

**Resolution**:
- Plumbed `api_key` from the UI through `configure_orchestrator` -> `create_orchestrator` -> `MagenticOrchestrator`.
- Magentic agents now correctly use the user-provided OpenAI key.

## Bug 4: Singleton EmbeddingService Causes Cross-Session Pollution (P0) - FIXED

**Resolution**:
- Removed the singleton pattern for `EmbeddingService`.
- Each request now gets a fresh `EmbeddingService` with a unique, isolated ChromaDB collection (`evidence_{uuid}`).
- `SentenceTransformer` model is lazily cached globally to maintain performance.

---

## Verification

Run `make check` to verify all tests pass.