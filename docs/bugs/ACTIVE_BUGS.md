# Active Bugs

> Last updated: 2025-11-28

## P0 - Critical

### Magentic Mode Report Generation
**File**: [FIX_PLAN_MAGENTIC_MODE.md](./FIX_PLAN_MAGENTIC_MODE.md)

**Symptom**: Magentic mode returns `ChatMessage` object instead of synthesized report text.

**Root Cause**:
- `event.message.text` extraction fails in orchestrator
- `max_rounds=3` too low for SearchAgent + JudgeAgent + ReportAgent sequence

**Workaround**: Use Simple mode (default) - works correctly with all LLM providers.

**Status**: Fix plan documented, not yet implemented.

---

## P1 - Minor UX

### Gradio Settings Accordion Won't Collapse
**File**: [P1_GRADIO_SETTINGS_CLEANUP.md](./P1_GRADIO_SETTINGS_CLEANUP.md)

**Symptom**: Settings accordion stays open after user interaction.

**Root Cause**: Nested `gr.Blocks` context prevents accordion state management.

**Impact**: UX only - all functionality works correctly.

**Status**: Solution documented, not yet implemented.

---

## Resolved Bugs

*None currently - bugs above are still open.*
