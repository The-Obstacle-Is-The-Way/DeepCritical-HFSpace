# P2 Bug: Duplicate Report Content in Output

**Date**: 2025-12-03
**Status**: OPEN
**Severity**: P2 (UX - Duplicate content confuses users)
**Component**: `src/orchestrators/advanced.py`
**Affects**: Both Free Tier (HuggingFace) AND Paid Tier (OpenAI)

---

## Executive Summary

This is a **confirmed stack bug**, NOT a model limitation. The duplicate report appears because:

1. Streaming events yield the full report content character-by-character
2. Final events (`MagenticFinalResultEvent`/`WorkflowOutputEvent`) contain the SAME content
3. No deduplication exists between streamed content and final event content
4. Both are appended to the output

---

## Symptom

The final research report appears **twice** in the UI output:
1. First as streaming content (with `📡 **STREAMING**:` prefix)
2. Then again as a complete event (without prefix)

---

## Root Cause

The `_process_event()` method handles final events but has **no access to buffer state**. The buffer was already cleared at line 337 before these events arrive.

```python
# Line 337: Buffer cleared
current_message_buffer = ""
continue

# Line 341: Final events processed WITHOUT buffer context
agent_event = self._process_event(event, iteration)  # No buffer info!
```

---

## The Fix (Consensus: Stateful Orchestrator Logic)

**Location**: `src/orchestrators/advanced.py` `run()` method

**Strategy**: Handle final events **inline in the run() loop** where buffer state exists. Track streaming volume to decide whether to re-emit content.

### Why This Is Correct

| Rejected Approach | Why Wrong |
|-------------------|-----------|
| UI-side string comparison | Wrong layer, fragile, treats symptom |
| Stateless `_process_event` fix | No state = can't know if streaming occurred |
| **Stateful run() loop** | ✅ Only place with full lifecycle visibility |

The `run()` loop is the **single source of truth** for the request lifecycle. It "saw" the content stream out. It must decide whether to re-emit.

### Implementation

```python
# In run() method, add tracking variable after line 302:
last_streamed_length: int = 0

# Before clearing buffer at line 337, save its length:
last_streamed_length = len(current_message_buffer)
current_message_buffer = ""
continue

# Replace lines 340-345 with inline handling of final events:
if isinstance(event, (MagenticFinalResultEvent, WorkflowOutputEvent)):
    final_event_received = True

    # DECISION: Did we stream substantial content?
    if last_streamed_length > 100:
        # YES: Final event is a SIGNAL, not a payload
        yield AgentEvent(
            type="complete",
            message="Research complete.",
            data={"iterations": iteration, "streamed_chars": last_streamed_length},
            iteration=iteration,
        )
    else:
        # NO: Final event must carry the payload (tool-only turn, cache hit)
        if isinstance(event, MagenticFinalResultEvent):
            text = self._extract_text(event.message) if event.message else "No result"
        else:  # WorkflowOutputEvent
            text = self._extract_text(event.data) if event.data else "Research complete"
        yield AgentEvent(
            type="complete",
            message=text,
            data={"iterations": iteration},
            iteration=iteration,
        )
    continue

# Keep existing fallback for other events:
agent_event = self._process_event(event, iteration)
```

### Why Threshold of 100 Chars?

- `> 0` is too aggressive (might catch single-word streams)
- `> 500` is too conservative (might miss short but complete responses)
- `> 100` distinguishes "real content was streamed" from "just status messages"

---

## Edge Cases Handled

| Scenario | `last_streamed_length` | Action |
|----------|------------------------|--------|
| Normal streaming report | 5000+ | Emit "Research complete." |
| Tool call, no text | 0 | Emit full content from final event |
| Very short response | 50 | Emit full content (fallback) |
| Agent switch mid-stream | Reset on switch | Tracks only final agent |

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/orchestrators/advanced.py` | 296-345 | Add `last_streamed_length`, handle final events inline |
| `src/orchestrators/advanced.py` | 532-552 | Optional: remove dead code from `_process_event()` |

---

## Test Plan

1. **Happy Path**: Run query, verify report appears ONCE
2. **Fallback**: Mock tool-only turn (no streaming), verify full content emitted
3. **Both Tiers**: Test Free Tier and Paid Tier

---

## Validation

This fix was independently validated by two AI agents (Claude and Gemini) analyzing the architecture. Both concluded:

> "The Stateful Orchestrator Fix is the correct engineering solution. The 'Source of Truth' is the Orchestrator's runtime state."

---

## Related

- **Not related to model quality** - This is a stack bug
- P1 Free Tier fix enabled streaming, exposing this bug
- SPEC-17 Accumulator Pattern addressed repr bug but created this side effect