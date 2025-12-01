# Active Bugs

> Last updated: 2025-12-01 (16:15 PST)
>
> **Note:** Completed bug docs archived to `docs/bugs/archive/`
> **See also:** [Code Quality Audit Findings (2025-11-30)](AUDIT_FINDINGS_2025_11_30.md)

## P3 - UX Polish

### P3 - Progress Bar Positioning/Overlap in ChatInterface
**File:** `docs/bugs/P3_PROGRESS_BAR_POSITIONING.md`
**Found:** 2025-12-01 (HuggingFace Spaces)

**Problem:** `gr.Progress()` bar renders in strange position inside ChatInterface - floats mid-chat, overlaps text.

**Root Cause:** Mixing two progress mechanisms:
1. `gr.Progress()` - general purpose, not designed for ChatInterface
2. `ChatInterface.show_progress` - built-in chat progress

**Recommended Fix:** Remove `gr.Progress()`, rely on emoji status text we already emit. Low priority - UX polish only.

---

## P2 - UX Friction

### P2 - Advanced Mode Cold Start Has No User Feedback (✅ FIXED)
**File:** `docs/bugs/P2_ADVANCED_MODE_COLD_START_NO_FEEDBACK.md`
**Issue:** [#108](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/108)
**Found:** 2025-12-01 (Gradio Testing)

**Problem:** Three "dead zones" with no visual feedback during Advanced Mode startup:
1. **Dead Zone #1** (5-15s): Between STARTED → THINKING ✅ FIXED (granular events)
2. **Dead Zone #2** (10-30s): Between THINKING → PROGRESS (first LLM call) ✅ FIXED (Progress Bar)
3. **Dead Zone #3** (30-90s): After PROGRESS (SearchAgent executing) ✅ FIXED (Pre-warming + Progress Bar)

**Phase 1 Fix (commit dbf888c):**
- Added granular progress events during initialization
- Users now see "Loading embedding service...", "Initializing research memory...", "Building agent team..."
- Significantly improves perceived responsiveness

**Phase 2/3 Fix (Latest):**
- Implemented service pre-warming (`service_loader.warmup_services`)
- Added native Gradio progress bar (`gr.Progress`) to `research_agent`
- Visual feedback is now continuous throughout the entire lifecycle

---

## P1 - Important

### P1 - Memory Layer Not Integrated (Post-Hackathon)
**Issue:** [#73](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/73)
**Spec:** [SPEC_08_INTEGRATE_MEMORY_LAYER.md](../specs/SPEC_08_INTEGRATE_MEMORY_LAYER.md)

**Problem:** Structured memory (hypotheses, conflicts) is isolated in "God Mode" only.
**Solution:** Extract memory into shared service, integrate into Simple and Advanced modes.
**Status:** Spec written. Blocked until post-hackathon.

---

## Resolved Bugs

### ~~P1 - HuggingFace Router 401 Unauthorized~~ FIXED
**File:** `docs/bugs/P1_HUGGINGFACE_ROUTER_401_HYPERBOLIC.md`
**Found:** 2025-12-01
**Resolved:** 2025-12-01

- Problem: 401 errors from HuggingFace Router (Hyperbolic/Novita providers)
- **Actual Root Cause:** HF_TOKEN in `.env` and Spaces secrets was **invalid/expired**
- Fix: Generated new valid HF_TOKEN, updated `.env` and Spaces secrets
- Also switched default model to `Qwen/Qwen2.5-72B-Instruct` for better reliability

### ~~P0 - Simple Mode Ignores Forced Synthesis~~ FIXED
**File:** `docs/bugs/P0_SIMPLE_MODE_FORCED_SYNTHESIS_BYPASS.md`
**Issue:** [#113](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/113)
**PR:** [#115](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/pull/115) (SPEC-16)
**Found:** 2025-12-01
**Resolved:** 2025-12-01

- Problem: Simple Mode ignored forced synthesis signals from Judge.
- Fix: SPEC-16 unified architecture - removed Simple Mode entirely, integrated HuggingFace into Advanced Mode.
- Simple Mode code deleted, capability preserved via `HuggingFaceChatClient` adapter.

### ~~P1 - Advanced Mode Exposes Uninterpretable Chain-of-Thought~~ FIXED
**File:** `docs/bugs/P1_ADVANCED_MODE_UNINTERPRETABLE_CHAIN_OF_THOUGHT.md`
**PR:** [#107](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/pull/107)
**Found:** 2025-12-01
**Resolved:** 2025-12-01

- Problem: Advanced mode exposed raw `task_ledger` and `instruction` events, truncated mid-word.
- Fix: Filtered internal events, transformed `user_task` to progress type, smart sentence-aware truncation.
- Tests: `tests/unit/orchestrators/test_advanced_events.py` (5 tests)
- CodeRabbit review addressed: test markers, edge case handling, truncation test coverage.

### ~~P0 - Advanced Mode Timeout Yields No Synthesis~~ FIXED
**File:** `docs/bugs/P0_ADVANCED_MODE_TIMEOUT_NO_SYNTHESIS.md`
**Found:** 2025-11-30 (Manual Testing)
**Resolved:** 2025-12-01

- Problem: Advanced mode timed out and displayed "Synthesizing..." but no synthesis occurred.
- Root Causes:
  1. Timeout handler yielded misleading message without calling ReportAgent
  2. Factory used wrong setting (`max_iterations=10` instead of `advanced_max_rounds=5`)
  3. Missing `get_context_summary()` in ResearchMemory
- Fix:
  1. Implemented actual synthesis on timeout via ReportAgent invocation
  2. Factory now uses `settings.advanced_max_rounds` (5)
  3. Added `get_context_summary()` to ResearchMemory
- Tests: `tests/unit/orchestrators/test_advanced_timeout.py`
- Key files: `src/orchestrators/advanced.py`, `src/orchestrators/factory.py`, `src/services/research_memory.py`

### ~~P0 - Free Tier Synthesis Incorrectly Uses Server-Side API Keys~~ FIXED
**File:** `docs/bugs/P1_SYNTHESIS_BROKEN_KEY_FALLBACK.md`
**PR:** [#103](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/pull/103)
**Found:** 2025-11-30 (Testing)
**Resolved:** 2025-11-30
**Verified:** Free Tier now produces full LLM-synthesized research reports ✅

- Problem: Simple Mode crashed with "OpenAIError" on HuggingFace Spaces when user provided no key but admin key was invalid.
- Root Cause: Synthesis logic bypassed the Free Tier judge and incorrectly used server-side keys via `get_model()`.
- Fix: Implemented `synthesize()` in `HFInferenceJudgeHandler` to use free HuggingFace Inference, ensuring consistency with the judge phase.
- Key files: `src/agent_factory/judges.py`, `src/orchestrators/simple.py`

### ~~P0 - Synthesis Fails with OpenAIError in Free Mode~~ FIXED
**File:** `docs/bugs/P0_SYNTHESIS_PROVIDER_MISMATCH.md`
**Found:** 2025-11-30 (Code Audit)
**Resolved:** 2025-11-30

- Problem: "Simple Mode" (Free Tier) crashed with `OpenAIError`.
- Root Cause: `get_model()` defaulted to OpenAI regardless of available keys.
- Fix: Implemented auto-detection in `judges.py` (OpenAI > Anthropic > HuggingFace).
- Added extensive unit tests and regression tests.

### ~~P0 - Simple Mode Never Synthesizes~~ FIXED
**PR:** [#71](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/pull/71) (SPEC_06)
**Commit**: `5cac97d` (2025-11-29)

- Root cause: LLM-as-Judge recommendations were being IGNORED
- Fix: Code-enforced termination criteria (`_should_synthesize()`)
- Added combined score thresholds, late-iteration logic, emergency fallback
- Simple mode now synthesizes instead of spinning forever

### ~~P3 - Magentic Mode Missing Termination Guarantee~~ FIXED
**Commit**: `d36ce3c` (2025-11-29)

- Added `final_event_received` tracking in `orchestrator_magentic.py`
- Added fallback yield for "max iterations reached" scenario
- Verified with `test_magentic_termination.py`

### ~~P0 - Magentic Mode Report Generation~~ FIXED
**Commit**: `9006d69` (2025-11-29)

- Fixed `_extract_text()` to handle various message object formats
- Increased `max_rounds=10` (was 3)
- Added `temperature=1.0` for reasoning model compatibility
- Advanced mode now produces full research reports

### ~~P1 - Streaming Spam + API Key Persistence~~ FIXED
**Commit**: `0c9be4a` (2025-11-29)

- Streaming events now buffered (not token-by-token spam)
- API key persists across example clicks via `gr.State`
- Examples use explicit `None` values to avoid overwriting keys

### ~~P2 - Missing "Thinking" State~~ FIXED
**Commit**: `9006d69` (2025-11-29)

- Added `"thinking"` event type with hourglass icon
- Yields "Multi-agent reasoning in progress..." before blocking workflow call
- Users now see feedback during 2-5 minute initial processing

### ~~P2 - Gradio Example Not Filling Chat Box~~ FIXED
**Commit**: `2ea01fd` (2025-11-29)

- Third example (HSDD) wasn't populating chat box when clicked
- Root cause: Parentheses in `HSDD (Hypoactive Sexual Desire Disorder)`
- Fix: Simplified to `Testosterone therapy for Hypoactive Sexual Desire Disorder?`

### ~~P1 - Gradio Settings Accordion~~ WONTFIX

Decision: Removed nested Blocks, using ChatInterface directly.
Accordion behavior is default Gradio - acceptable for demo.

---

## How to Report Bugs

1. Create `docs/bugs/P{N}_{SHORT_NAME}.md`
2. Include: Symptom, Root Cause, Fix Plan, Test Plan
3. Update this index
4. Priority: P0=blocker, P1=important, P2=UX, P3=edge case
