# Senior Agent Audit Results: DeepBoner Codebase

**Date**: 2025-11-28
**Auditor**: Claude (Senior Software Engineer)
**Status**: COMPLETE

---

## Executive Summary

The DeepBoner codebase has **4 critical defects** that render the demo non-functional for most users. The most severe is a **data leak** where the vector database persists across user sessions, causing search result corruption and potential privacy issues. Additionally, the "Advanced" mode ignores user-provided API keys, and the "Free Tier" mode fails silently when quotas are exhausted.

**Recommendation**: Immediate remediation of P0 bugs is required before hackathon submission.

---

## 1. Verification of Known Bugs (P0_CRITICAL_BUGS.md)

| Bug | Claim | Verification Status | Notes |
| :--- | :--- | :--- | :--- |
| **Bug 1** | Free Tier LLM Quota Exhausted | **CONFIRMED** | `HFInferenceJudgeHandler` catches errors but returns a fallback assessment with `recommendation="continue"`. This causes the orchestrator to loop uselessly until `max_iterations` is reached. The user sees no error message. |
| **Bug 2** | Evidence Counter Shows 0 | **CONFIRMED** | Directly caused by Bug 4. Deduplication logic works correctly *in isolation*, but fails because the underlying ChromaDB collection is polluted with stale data from previous sessions. |
| **Bug 3** | API Key Not Passed to Advanced | **CONFIRMED** | `create_orchestrator` in `orchestrator_factory.py` ignores the user's API key. `MagenticOrchestrator` and its agents fall back to `settings.openai_api_key` (env var), which is empty for BYOK users. |
| **Bug 4** | Singleton EmbeddingService | **CONFIRMED** | `EmbeddingService` is a global singleton with an in-memory ChromaDB. The collection is never cleared. Data leaks between sessions, causing valid new results to be marked as duplicates of old results. |

---

## 2. New Bugs Found

### Bug 5: Search Error Swallowing (P2)
**File**: `src/orchestrator.py` / `src/tools/search_handler.py`
**Symptoms**: If all search tools fail (e.g., network issue, API limit), the UI shows "Found 0 sources" without explaining why.
**Root Cause**: `SearchHandler` captures exceptions and returns them in an `errors` list, but `Orchestrator` only logs them to the console (`logger.warning`) and proceeds with empty evidence.
**Fix**: Yield an `AgentEvent(type="error")` or include errors in the `search_complete` event message.

### Bug 6: Hardcoded Model Names (P3)
**File**: `src/agent_factory/judges.py`
**Symptoms**: Maintenance burden.
**Root Cause**: Model names like `meta-llama/Llama-3.1-8B-Instruct` are hardcoded in the class `HFInferenceJudgeHandler` rather than pulled from `config.py`.
**Fix**: Move to `Settings`.

---

## 3. Code Quality Concerns

1.  **Singleton Abuse**: The `_embedding_service` global in `src/services/embeddings.py` is a major architectural flaw for a multi-user web app (even a demo). It should be scoped to the `Orchestrator` instance.
2.  **Inconsistent Factory Signatures**: `create_orchestrator` does not accept `api_key`, forcing hacks or reliance on global env vars.
3.  **Silent Failures**: The pervasive use of `try...except Exception` with only logging (no user feedback) makes debugging difficult for end-users.

---

## 4. Recommended Fix Order

### Step 1: Fix the Data Leak (Bug 4 & 2)
**Why**: Prevents result corruption and cross-user data leakage.
**Plan**:
1.  Remove singleton pattern from `src/services/embeddings.py`.
2.  Make `EmbeddingService` an instance variable of `Orchestrator`.
3.  Initialize a fresh `EmbeddingService` (and ChromaDB collection) for each `run()`.

### Step 2: Fix Advanced Mode BYOK (Bug 3)
**Why**: Enables the core "Advanced" feature for judges/users.
**Plan**:
1.  Update `create_orchestrator` signature to accept `api_key`.
2.  Update `MagenticOrchestrator` to accept `api_key`.
3.  Update `configure_orchestrator` in `app.py` to pass the key.
4.  Ensure `MagenticOrchestrator` constructs `OpenAIChatClient` with the user's key.

### Step 3: Fix Free Tier Experience (Bug 1)
**Why**: Ensures a usable fallback for those without keys.
**Plan**:
1.  In `HFInferenceJudgeHandler`, detect 402/429 errors.
2.  If caught, return a `JudgeAssessment` that triggers a "Complete" event with a clear error message, rather than "Continue".
3.  Add `HF_TOKEN` to the deployment environment if possible.

---

## Verification Plan

After applying fixes, run:
1.  **Unit Tests**: `make check`
2.  **Manual Test (Simple)**: Run without key, verify 402 error is handled OR works if token added.
3.  **Manual Test (Advanced)**: Run with OpenAI key, verify it proceeds past initialization.
4.  **Manual Test (Dedup)**: Run same query twice. Second run should find same number of results (not 0).
