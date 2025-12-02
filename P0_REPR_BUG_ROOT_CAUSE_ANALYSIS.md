# P0: Event Handling Implementation Spec

**Status**: FIXED
**Priority**: P0
**Source of Truth**: `reference_repos/microsoft-agent-framework/python/samples/autogen-migration/orchestrations/04_magentic_one.py`

---

## Root Cause (One Sentence)

We were extracting content from `MagenticAgentMessageEvent.message` — **the wrong event type** — instead of using `MagenticAgentDeltaEvent.text` as the sole source of streaming content.

---

## The Fix: Correct Event Handling Per Microsoft SSOT

| Event Type | Correct Usage | What We Were Doing (Wrong) |
|------------|---------------|----------------------------|
| `MagenticAgentDeltaEvent` | **Extract `.text`** - This is the ONLY source of content | Partially used, not accumulated |
| `MagenticAgentMessageEvent` | **Signal only** - Agent turn complete. IGNORE `.message` | Extracting `.message.text` (hits repr bug) |
| `MagenticFinalResultEvent` | **Extract `.message.text`** - Final synthesis result | Correct |

---

## Implementation: Accumulator Pattern

From Microsoft's `04_magentic_one.py` (lines 108-138):

```python
# Microsoft's Pattern
async for event in workflow.run_stream(task):
    if isinstance(event, MagenticAgentDeltaEvent):
        # STREAM CONTENT: Accumulate and display
        if event.text:
            print(event.text, end="", flush=True)

    elif isinstance(event, MagenticAgentMessageEvent):
        # SIGNAL ONLY: Agent done. Print newline. DO NOT read .message
        print()

    elif isinstance(event, MagenticFinalResultEvent):
        # FINAL RESULT: Safe to read .message.text
        print(event.message.text)
```

---

## Our Implementation (`src/orchestrators/advanced.py`)

**Status**: ✅ IMPLEMENTED (lines 241-308)

```python
# 1. Accumulate streaming content (ONLY source of truth)
if isinstance(event, MagenticAgentDeltaEvent):
    if event.text:
        current_message_buffer += event.text
        yield AgentEvent(type="streaming", message=event.text, ...)

# 2. Use buffer on completion signal (IGNORE event.message)
if isinstance(event, MagenticAgentMessageEvent):
    text_content = current_message_buffer or "Action completed (Tool Call)"
    yield AgentEvent(message=f"{agent_name}: {text_content[:200]}...", ...)
    current_message_buffer = ""  # Reset for next agent

# 3. Final result - safe to extract
if isinstance(event, MagenticFinalResultEvent):
    text = self._extract_text(event.message)
    yield AgentEvent(type="complete", message=text, ...)
```

---

## Why This Eliminates the Repr Bug

The repr bug occurs at `_magentic.py:1730`:

```python
text = last.text or str(last)  # Falls back to repr() for tool-only messages
```

By **never reading** `MagenticAgentMessageEvent.message.text`, we never hit this code path.

**The repr bug is eliminated by correct implementation — no upstream fix required.**

---

## Verification Checklist

- [x] `MagenticAgentDeltaEvent.text` used as sole content source
- [x] `MagenticAgentMessageEvent` used as signal only (buffer consumed, not `.message`)
- [x] `MagenticFinalResultEvent.message.text` extracted for final result
- [x] Buffer reset on agent switch and completion
- [x] Remove dead code path in `_process_event()` that still calls `_extract_text` on `MagenticAgentMessageEvent`

---

## Remaining Cleanup

✅ **DONE** - Dead code paths for `MagenticAgentMessageEvent` and `MagenticAgentDeltaEvent` have been removed from `_process_event()`. Comments now explain these events are handled by the Accumulator Pattern in `run()`.
