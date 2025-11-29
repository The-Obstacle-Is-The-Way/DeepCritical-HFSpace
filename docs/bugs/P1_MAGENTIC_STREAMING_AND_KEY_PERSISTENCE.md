# Bug Report: Magentic Mode Integration Issues

## Status
- **Date:** 2025-11-29
- **Reporter:** CLI User
- **Priority:** P1 (UX Degradation + Deprecation Warnings)
- **Component:** `src/app.py`, `src/orchestrator_magentic.py`, `src/utils/llm_factory.py`

---

## Bug 1: Token-by-Token Streaming Spam

### Symptoms
When running Magentic (Advanced) mode, the UI shows hundreds of individual lines like:
```
📡 STREAMING: Below
📡 STREAMING: is
📡 STREAMING: a
📡 STREAMING: curated
📡 STREAMING: list
...
```

Each token is displayed as a separate streaming event, creating visual spam and making it impossible to read the output until completion.

### Root Cause
**File:** `src/orchestrator_magentic.py:247-254`

```python
elif isinstance(event, MagenticAgentDeltaEvent):
    if event.text:
        return AgentEvent(
            type="streaming",
            message=event.text,  # Single token!
            data={"agent_id": event.agent_id},
            iteration=iteration,
        )
```

Every LLM token emits a `MagenticAgentDeltaEvent`, which creates an `AgentEvent(type="streaming")`.

**File:** `src/app.py:170-180`

```python
async for event in orchestrator.run(message):
    event_md = event.to_markdown()
    response_parts.append(event_md)  # Appends EVERY token

    if event.type == "complete":
        yield event.message
    else:
        yield "\n\n".join(response_parts)  # Yields ALL accumulated tokens
```

For N tokens, this yields N times, each time showing all previous tokens. This is O(N²) string operations and creates massive visual spam.

### Proposed Fix Options

**Option A: Buffer streaming tokens (recommended)**
```python
# In app.py - accumulate streaming tokens, yield periodically
streaming_buffer = ""
last_yield_time = time.time()

async for event in orchestrator.run(message):
    if event.type == "streaming":
        streaming_buffer += event.message
        # Only yield every 500ms or on newline
        if time.time() - last_yield_time > 0.5 or "\n" in event.message:
            yield f"📡 {streaming_buffer}"
            last_yield_time = time.time()
    elif event.type == "complete":
        yield event.message
    else:
        # Non-streaming events
        response_parts.append(event.to_markdown())
        yield "\n\n".join(response_parts)
```

**Option B: Don't yield streaming events at all**
```python
# In app.py - only yield meaningful events
async for event in orchestrator.run(message):
    if event.type == "streaming":
        continue  # Skip token-by-token spam
    # ... rest of logic
```

**Option C: Fix at orchestrator level**
Don't emit `AgentEvent` for every delta - buffer in `_process_event`.

---

## Bug 2: API Key Does Not Persist in Textbox

### Symptoms
1. User opens the "Mode & API Key" accordion
2. User pastes their API key into the password textbox
3. User clicks an example OR clicks elsewhere
4. The API key textbox is now empty - value lost

### Root Cause
**File:** `src/app.py:223-237`

```python
additional_inputs_accordion=additional_inputs_accordion,
additional_inputs=[
    gr.Radio(...),
    gr.Textbox(
        label="🔑 API Key (Optional)",
        type="password",
        # No `value` parameter - defaults to empty
        # No state persistence mechanism
    ),
],
```

Gradio's `ChatInterface` with `additional_inputs` has known issues:
1. Clicking examples resets additional inputs to defaults
2. The accordion state and input values may not persist correctly
3. No explicit state management for the API key

### Proposed Fix Options

**Option A: Use `gr.State` for persistence**
```python
api_key_state = gr.State("")

def research_agent(message, history, mode, api_key, api_key_state):
    # Use api_key_state if api_key is empty
    effective_key = api_key or api_key_state
    ...
    return response, effective_key  # Return to update state
```

**Option B: Use browser localStorage via JavaScript**
```python
demo.load(js="""
    () => {
        const saved = localStorage.getItem('deepboner_api_key');
        if (saved) document.querySelector('input[type=password]').value = saved;
    }
""")
```

**Option C: Environment variable only (remove BYOK textbox)**
Remove the API key input entirely. Require users to set `OPENAI_API_KEY` in HuggingFace Secrets. This is more secure but less user-friendly.

**Option D: Use Gradio LoginButton or HuggingFace OAuth**
Leverage HF's built-in auth and secrets management.

---

## Bug 3: Deprecated `OpenAIModel` Import

### Symptoms
HuggingFace Spaces logs show deprecation warning:
```
DeprecationWarning: OpenAIModel is deprecated, use OpenAIChatModel instead
```

### Root Cause
**Files using deprecated API:**
- `src/app.py:9` - `from pydantic_ai.models.openai import OpenAIModel`
- `src/utils/llm_factory.py:59` - `from pydantic_ai.models.openai import OpenAIModel`

**File already using correct API:**
- `src/agent_factory/judges.py:12` - `from pydantic_ai.models.openai import OpenAIChatModel`

### Fix
Replace all `OpenAIModel` imports with `OpenAIChatModel`:

```python
# Before (deprecated)
from pydantic_ai.models.openai import OpenAIModel
model = OpenAIModel(settings.openai_model, provider=provider)

# After (correct)
from pydantic_ai.models.openai import OpenAIChatModel
model = OpenAIChatModel(settings.openai_model, provider=provider)
```

**Files to update:**
1. `src/app.py` - lines 9, 64, 73
2. `src/utils/llm_factory.py` - lines 59, 67

---

## Bug 4: Asyncio Event Loop Garbage Collection Error

### Symptoms
HuggingFace Spaces logs show intermittent errors:
```
ValueError: Invalid file descriptor: -1
Exception ignored in: <function BaseSelector.__del__ at 0x...>
```

### Root Cause
This occurs during garbage collection of asyncio event loops. Likely causes:
1. Event loop cleanup timing issues in Gradio's threaded model
2. Selector objects being garbage-collected before proper cleanup
3. Concurrent access to event loop resources during shutdown

### Analysis
The codebase uses `asyncio.get_running_loop()` correctly (not the deprecated `get_event_loop()`).
This error appears to be a Gradio/HuggingFace Spaces environment issue rather than a code bug.

### Potential Mitigations
1. **Add explicit cleanup**: Use `asyncio.get_event_loop().close()` in appropriate places
2. **Ignore in logs**: This is a known Python issue and can be safely ignored if it doesn't affect functionality
3. **File issue with Gradio**: If reproducible, report to Gradio GitHub

### Impact
- **Severity**: Low - appears to be a cosmetic log issue
- **User-visible**: No - errors occur during garbage collection, not during request handling

---

## Recommended Priority

1. **Bug 1 (Streaming Spam)**: HIGH - makes Advanced mode unusable for reading output
2. **Bug 3 (OpenAIModel deprecation)**: MEDIUM - fix to avoid future breakage
3. **Bug 2 (Key Persistence)**: LOW - annoying but users can re-paste
4. **Bug 4 (Asyncio GC)**: LOW - cosmetic log noise, monitor but likely no action needed

## Testing Plan

1. Run Advanced mode query, verify no token-by-token spam
2. Verify no deprecation warnings in logs after OpenAIChatModel fix
3. Paste API key, click example, verify key persists
4. Refresh page, verify key persists (if using localStorage)
5. Run `make check` - all tests pass
