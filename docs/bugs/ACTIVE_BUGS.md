# Active Bugs

> Last updated: 2025-12-03
>
> **Note:** Completed bug docs archived to `docs/bugs/archive/`
> **See also:** [ARCHITECTURE.md](../ARCHITECTURE.md) for unified architecture plan

---

## Currently Active Bugs

### P3 - Progress Bar Positioning in ChatInterface

**File:** `docs/bugs/P3_PROGRESS_BAR_POSITIONING.md`
**Status:** OPEN - Low Priority UX Polish

**Problem:** The `gr.Progress()` bar renders in a strange position when used inside ChatInterface, causing visual overlap with chat messages.

**Recommended Fix:** Remove `gr.Progress()` entirely and rely on emoji status messages in chat output.

---

## Tech Debt (Future Roadmap)

### P3 - Remove Anthropic Partial Wiring

**File:** `docs/future-roadmap/P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md`
**Status:** OPEN - Tech Debt

**Problem:** Anthropic is partially wired but NOT fully supported (no embeddings API). Creates confusion.

**Fix:** Remove all Anthropic references from codebase. See doc for file list.

---

## Resolved Bugs (December 2025)

All resolved bugs have been moved to `docs/bugs/archive/`. Summary:

### P0 Bugs (All FIXED)
- **P0 Repr Bug** - FIXED in PR #117 via Accumulator Pattern
- **P0 AIFunction Not JSON Serializable** - FIXED, full tool support for HuggingFace
- **P0 HuggingFace Tool Calling Broken** - FIXED, history serialization + Accumulator Pattern
- **P0 Simple Mode Forced Synthesis Bypass** - N/A, simple.py deleted (Unified Architecture)
- **P0 Synthesis Provider Mismatch** - FIXED, auto-detect in judges.py
- **P0 Advanced Mode Timeout No Synthesis** - FIXED, actual synthesis on timeout

### P1 Bugs (All FIXED)
- **P1 Free Tier Tool Execution Failure** - FIXED in PR fix/P1-free-tier-tool-execution, removed premature marker
- **P1 Gradio Example Click Auto-Submits** - FIXED in PR #120, prevents auto-submit on example click
- **P1 HuggingFace Router 401 Hyperbolic** - FIXED, invalid token was root cause
- **P1 HuggingFace Novita 500 Error** - SUPERSEDED, switched to 7B model
- **P1 Advanced Mode Uninterpretable Chain-of-Thought** - FIXED in PR #107
- **P1 Synthesis Broken Key Fallback** - FIXED in PR #103
- **P1 Simple Mode Removed Breaks Free Tier UX** - FIXED via Accumulator Pattern (PR #117)

### P2 Bugs (All FIXED)
- **P2 7B Model Garbage Output** - SUPERSEDED by P1 Free Tier fix (root cause was premature marker, not model capacity)
- **P2 Advanced Mode Cold Start No Feedback** - FIXED, all phases complete
- **P2 Architectural BYOK Gaps** - FIXED, end-to-end BYOK support in PR #119

---

## How to Report Bugs

1. Create `docs/bugs/P{N}_{SHORT_NAME}.md`
2. Include: Symptom, Root Cause, Fix Plan, Test Plan
3. Update this index
4. Priority: P0=blocker, P1=important, P2=UX, P3=edge case/tech debt

---

## Archived Documentation

The following have been moved to `docs/bugs/archive/`:
- All resolved P0-P2 bug reports
- Code quality audit findings (2025-11-30)
- Gradio example vs chat arrow analysis

Additional documentation moved:
- `HF_FREE_TIER_ANALYSIS.md` → `docs/architecture/`
- `TOOL_ANALYSIS_CRITICAL.md` → `docs/future-roadmap/`
- `P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md` → `docs/future-roadmap/`
