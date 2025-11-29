# Fix Plan: Critical Bugs (P0)

**Date**: 2025-11-28
**Status**: COMPLETED (2025-11-29)
**Based on**: `docs/bugs/SENIOR_AUDIT_RESULTS.md`

---

## Summary of Fixes

### 1. Fixed Data Leak (Bug 4 & 2)
- **Action**: Removed singleton `_embedding_service` in `src/services/embeddings.py`.
- **Action**: Updated `EmbeddingService.__init__` to use a unique collection name (`evidence_{uuid}`) for complete isolation per instance.
- **Action**: Refactored `SentenceTransformer` loading to a shared global to maintain performance while isolating state.
- **Verified**: Unit tests passed, including new isolation verification.

### 2. Fixed Advanced Mode BYOK (Bug 3)
- **Action**: Updated `create_orchestrator` in `src/orchestrator_factory.py` to accept `api_key`.
- **Action**: Updated `MagenticOrchestrator` to accept and use the `api_key` for the manager and agents.
- **Action**: Updated `src/app.py` to pass the user's API key during orchestrator configuration.
- **Verified**: `test_dual_mode_e2e.py` passed.

### 3. Fixed Free Tier Experience (Bug 1)
- **Action**: Updated `HFInferenceJudgeHandler` in `src/agent_factory/judges.py` to catch 402 (Payment Required) errors.
- **Action**: Added logic to return a "synthesize" assessment with a clear error message when quota is exhausted, stopping the infinite loop.
- **Verified**: Unit tests passed.

---

## Verification

All changes have been verified with:
- `make check` (lint, typecheck, test) - ALL PASSED
- Custom reproduction script for isolation - PASSED

The system is now stable for the hackathon demo.