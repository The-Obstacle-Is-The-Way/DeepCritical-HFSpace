# Active Bugs

> Last updated: 2025-11-29

## P0 - Blocker

*(None - P0 bugs resolved)*

---

## P3 - Architecture/Enhancement

### P3 - Missing Structured Cognitive Memory
**File:** `P3_ARCHITECTURAL_GAP_STRUCTURED_MEMORY.md`
**Spec:** [SPEC_07_LANGGRAPH_MEMORY_ARCH.md](../specs/SPEC_07_LANGGRAPH_MEMORY_ARCH.md)

**Problem:** AdvancedOrchestrator uses chat-based state (context drift on long runs).
**Solution:** Implement LangGraph StateGraph with explicit hypothesis/conflict tracking.
**Status:** Spec complete, implementation pending.

### P3 - Ephemeral Memory (No Persistence)
**File:** `P3_ARCHITECTURAL_GAP_EPHEMERAL_MEMORY.md`

**Problem:** ChromaDB uses in-memory client despite `settings.chroma_db_path` existing.
**Solution:** Switch to `PersistentClient(path=settings.chroma_db_path)`.
**Status:** Quick fix identified, not yet implemented.

---

## Resolved Bugs

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
