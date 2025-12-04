# P2 Bug: Duplicate Report Content in Output

**Date**: 2025-12-03
**Status**: OPEN
**Severity**: P2 (UX - Duplicate content confuses users)
**Component**: `src/orchestrators/advanced.py` + `src/app.py`
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

Example:
```
📡 **STREAMING**:
### Summary of Drugs and Mechanisms of Action
...
### Conclusion
Post-menopausal women experiencing libido issues can benefit from...
### Recommendations
- Estrogen Therapy: Effective in enhancing...

Based on the information gathered, we have identified...   <-- DUPLICATE STARTS
### Summary of Drugs and Mechanisms of Action
...
### Conclusion
Post-menopausal women experiencing libido issues can benefit from...
### Recommendations
- Estrogen Therapy: Effective in enhancing...
```

---

## Root Cause Analysis

### Event Flow (Current - Buggy)

```
1. Reporter Agent streams content
   └─ MagenticAgentDeltaEvent × N
      └─ Each yields AgentEvent(type="streaming", message=delta)
      └─ app.py: streaming_buffer += event.message
      └─ User sees: "📡 **STREAMING**: [content building up]"

2. Reporter Agent completes
   └─ MagenticAgentMessageEvent
      └─ Yields truncated completion: "reporter: [first 200 chars]..."
      └─ app.py: flushes streaming_buffer to response_parts

3. Workflow ends
   └─ MagenticFinalResultEvent OR WorkflowOutputEvent
      └─ Contains FULL report content (same as streaming)
      └─ Yields AgentEvent(type="complete", message=FULL_CONTENT)
      └─ app.py: appends event.message to response_parts
      └─ User sees: [SAME CONTENT AGAIN]
```

### Bug Location

**`src/orchestrators/advanced.py` lines 532-552:**
```python
elif isinstance(event, MagenticFinalResultEvent):
    text = self._extract_text(event.message) if event.message else "No result"
    return AgentEvent(
        type="complete",
        message=text,  # <-- FULL content, already streamed
        ...
    )

elif isinstance(event, WorkflowOutputEvent):
    if event.data:
        text = self._extract_text(event.data)
        return AgentEvent(
            type="complete",
            message=text,  # <-- FULL content, already streamed
            ...
        )
```

**`src/app.py` lines 229-232:**
```python
if event.type == "complete":
    response_parts.append(event.message)  # <-- Appends duplicate
    yield "\n\n".join(response_parts)
```

### Why It Happens

1. **Streaming events** yield the full report character-by-character
2. **Final events** (`MagenticFinalResultEvent`, `WorkflowOutputEvent`) contain the same full content
3. **No deduplication** exists between streamed content and final event content
4. **app.py appends both** to the output

---

## Impact

| Aspect | Impact |
|--------|--------|
| UX | Report appears twice, looks buggy |
| Token usage | Renders same content twice |
| Trust | Users may think system is broken |

---

## Proposed Fix Options

### Option 1: Skip Complete Event if Content Matches Streaming (Recommended)

**Location**: `src/app.py` lines 229-232

```python
if event.type == "complete":
    # Skip if content matches what we already streamed
    streaming_content = next(
        (p.replace("📡 **STREAMING**: ", "") for p in response_parts if p.startswith("📡 **STREAMING**:")),
        None
    )
    if streaming_content and event.message.strip() == streaming_content.strip():
        continue  # Skip duplicate
    response_parts.append(event.message)
    yield "\n\n".join(response_parts)
```

**Pros**: Simple, targets exact issue
**Cons**: String comparison may be fragile

### Option 2: Track Streamed Content Hash

**Location**: `src/app.py`

```python
streaming_hash = None
...
if streaming_buffer:
    streaming_hash = hash(streaming_buffer.strip())
    response_parts.append(f"📡 **STREAMING**: {streaming_buffer}")
    streaming_buffer = ""
...
if event.type == "complete":
    if streaming_hash and hash(event.message.strip()) == streaming_hash:
        continue  # Skip duplicate
    response_parts.append(event.message)
```

**Pros**: More robust comparison
**Cons**: Hash collision possible (unlikely)

### Option 3: Don't Emit Complete Event Content from Orchestrator

**Location**: `src/orchestrators/advanced.py` lines 532-552

Replace full content with summary:
```python
elif isinstance(event, MagenticFinalResultEvent):
    return AgentEvent(
        type="complete",
        message="Research complete.",  # Don't repeat content
        data={"iterations": iteration},
        iteration=iteration,
    )
```

**Pros**: Clean separation of streaming vs completion
**Cons**: Loses fallback if streaming failed

### Option 4: Flag-Based Deduplication in Orchestrator

**Location**: `src/orchestrators/advanced.py`

Track if substantial streaming occurred:
```python
has_substantial_streaming = len(current_message_buffer) > 100

# In _process_event for final events:
if has_substantial_streaming:
    return AgentEvent(
        type="complete",
        message="Research complete.",  # Don't repeat
        ...
    )
```

---

## Recommended Fix

**Option 3** is cleanest - the orchestrator should not re-emit content that was already streamed.

**Implementation**:
1. Track `streamed_report_length` in the run loop
2. If substantial content was streamed (>500 chars), emit minimal complete message
3. If no streaming occurred, emit full content as fallback

---

## Files Involved

| File | Role |
|------|------|
| `src/orchestrators/advanced.py:532-552` | Emits duplicate complete events |
| `src/app.py:229-232` | Appends duplicate to output |

---

## Test Plan

1. Run Free Tier query: "What drugs improve female libido post-menopause?"
2. Verify report appears ONCE (with streaming prefix)
3. Verify `complete` event does NOT repeat content
4. Verify fallback works if streaming fails

---

## Deep Technical Analysis

### Microsoft Agent Framework Event Types

The framework emits these event types (all inherit from `WorkflowEvent`):

| Event Type | Purpose | Key Attributes |
|------------|---------|----------------|
| `MagenticAgentDeltaEvent` | Streaming tokens | `text`, `agent_id` |
| `MagenticAgentMessageEvent` | Agent turn complete | `message` (ChatMessage), `agent_id` |
| `MagenticFinalResultEvent` | Workflow final result | `message` (ChatMessage) |
| `MagenticOrchestratorMessageEvent` | Manager bookkeeping | `message`, `kind`, `orchestrator_id` |
| `WorkflowOutputEvent` | Workflow output | `data`, `source_executor_id` |

### Event Flow Trace

```
PHASE 1: Agent Streaming (Reporter)
─────────────────────────────────────
MagenticAgentDeltaEvent(text="##", agent_id="reporter")     → yields streaming event
MagenticAgentDeltaEvent(text=" Summary", agent_id="reporter") → yields streaming event
MagenticAgentDeltaEvent(text="\n", agent_id="reporter")     → yields streaming event
... (hundreds more delta events)
MagenticAgentDeltaEvent(text=".", agent_id="reporter")      → yields streaming event

→ Result: Full report content in streaming_buffer (app.py) and current_message_buffer (orchestrator)

PHASE 2: Agent Completion
─────────────────────────────────────
MagenticAgentMessageEvent(message=ChatMessage(...), agent_id="reporter")
→ _handle_completion_event() yields: "reporter: [first 200 chars]..."
→ Clears current_message_buffer
→ app.py flushes streaming_buffer to response_parts with "📡 **STREAMING**:" prefix

PHASE 3: Workflow Termination (THE BUG)
─────────────────────────────────────
MagenticFinalResultEvent(message=ChatMessage(...))  ← Contains SAME full report!
OR
WorkflowOutputEvent(data=ChatMessage(...))          ← Contains SAME full report!

→ _process_event() extracts text with _extract_text()
→ Returns AgentEvent(type="complete", message=FULL_REPORT)
→ app.py appends FULL_REPORT to response_parts (NO prefix)

RESULT: Report appears twice:
1. "📡 **STREAMING**: [full report]"
2. "[full report again]"
```

### Key Code Paths

**`advanced.py` lines 299-345 (main loop):**
```python
# Buffer is cleared HERE (line 337) after MagenticAgentMessageEvent
current_message_buffer = ""

# But MagenticFinalResultEvent comes AFTER and _process_event has no buffer context!
agent_event = self._process_event(event, iteration)  # line 341
if agent_event:
    yield agent_event  # line 345 - yields duplicate!
```

**`advanced.py` lines 532-539 (_process_event):**
```python
elif isinstance(event, MagenticFinalResultEvent):
    text = self._extract_text(event.message)  # Extracts FULL content
    return AgentEvent(type="complete", message=text)  # Returns FULL content
```

**`app.py` lines 229-232 (UI handling):**
```python
if event.type == "complete":
    response_parts.append(event.message)  # Appends to existing streamed content!
    yield "\n\n".join(response_parts)
```

### Why Buffer Clearing Doesn't Help

The `current_message_buffer` is cleared (line 337) BEFORE the final events arrive. So even if we wanted to compare, we've already lost the reference:

```python
# Line 327-338: Handle MagenticAgentMessageEvent
iteration += 1
comp_event, prog_event = self._handle_completion_event(...)
yield comp_event
yield prog_event
current_message_buffer = ""  # CLEARED!
continue

# Line 341-345: Handle final events (buffer is empty now!)
agent_event = self._process_event(event, iteration)  # No buffer context
```

### Potential Edge Cases

1. **Tool-only turns**: If agent makes tool calls without text, buffer is empty → fallback text used
2. **Multiple agents streaming**: Buffer clears on agent switch (line 311-313) → OK
3. **Timeout**: Uses `_handle_timeout()` which invokes ReportAgent directly → Different path
4. **No final event**: Falls back to "Research completed..." message (line 354-363) → OK

### Verification Needed

- [ ] Confirm `MagenticFinalResultEvent` vs `WorkflowOutputEvent` - which is emitted?
- [ ] Confirm bug occurs on both Free and Paid tiers
- [ ] Measure content length match between streaming and final event

---

## Related

- **Not related to model quality** - This is a stack bug, not model limitation
- P1 Free Tier fix (PR fix/P1-free-tier) enabled streaming, exposing this bug
- SPEC-17 Accumulator Pattern addressed repr bug but created this side effect
