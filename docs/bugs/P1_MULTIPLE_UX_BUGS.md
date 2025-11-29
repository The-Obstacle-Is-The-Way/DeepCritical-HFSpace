# P1 Bug Report: Multiple UX and Configuration Issues

## Status
- **Date:** 2025-11-29
- **Priority:** P1 (Multiple user-facing issues)
- **Components:** `src/app.py`, `src/orchestrator_magentic.py`

---

## Bug 1: API Key Cleared When Clicking Examples

### Symptoms
- User enters API key in textbox
- User clicks an example prompt
- API key textbox is cleared/reset

### Root Cause
Despite examples only having 2 columns `[message, mode]`, Gradio's ChatInterface still resets `additional_inputs` that aren't in the examples list. The comment on line 273-274 was incorrect:

```python
# API key persists because examples only include [message, mode] columns,
# so Gradio doesn't overwrite the api_key textbox when examples are clicked.
```

This assumption is **wrong** - Gradio resets ALL additional_inputs, not just those with example values.

### Potential Fix
Option A: Include API key column in examples (set to empty string explicitly)
```python
examples=[
    ["What drugs improve female libido?", "simple", ""],
    ...
]
```

Option B: Use JavaScript to preserve the value (hacky)

Option C: Move API key outside ChatInterface into a separate Blocks layout

### Research Needed
- Gradio ChatInterface 2025 behavior with partial examples
- Whether `cache_examples=False` affects this

---

## Bug 2: No Loading/Processing Indicator

### Symptoms
- User submits query
- UI shows "🚀 STARTED:" message but nothing else
- No spinner, no "thinking...", no indication work is happening
- User thinks it's frozen

### Container Logs Show
Work IS happening:
```
[info] Creating orchestrator mode=advanced
[info] Starting Magentic orchestrator query='...'
[info] Embedding service enabled
```

But user sees nothing for 30+ seconds.

### Root Cause
The Gradio ChatInterface doesn't show intermediate yields quickly enough, and we don't yield a "⏳ Processing..." message immediately.

### Proposed Fix
Add immediate feedback in `research_agent()`:
```python
yield "⏳ **Processing...** Searching PubMed, ClinicalTrials.gov, Europe PMC..."
```

---

## Bug 3: Advanced Mode Temperature Error

### Error
```
Unsupported value: 'temperature' does not support 0.3 with this model.
Only the default (1) value is supported.
```

### Root Cause
The `agent_framework` (Magentic) is using `temperature=0.3` but some OpenAI models (like `o3`, `o1`, reasoning models) only support `temperature=1`.

### Location
Likely in `src/orchestrator_magentic.py` or agent-framework configuration.

### Proposed Fix
1. Detect model type and skip temperature for reasoning models
2. Or: Remove explicit temperature setting, use model defaults
3. Or: Catch this error and fall back to default temperature

---

## Bug 4: HSDD Acronym Not Spelled Out

### Issue
Example prompt says:
```
"Evidence for testosterone therapy in women with HSDD?"
```

**HSDD = Hypoactive Sexual Desire Disorder** (low libido condition)

Most users (including doctors!) won't know this acronym.

### Fix
Change to:
```
"Evidence for testosterone therapy in women with HSDD (Hypoactive Sexual Desire Disorder)?"
```

Also update README if it uses this acronym.

---

## Bug 5: Free Tier Quota Exhausted (Expected Behavior)

### Logs
```
[error] HF Quota Exhausted error='402 Client Error: Payment Required...'
```

### This is NOT a bug
HuggingFace free tier has limited credits. When exhausted:
- User should enter their own API key
- The app correctly falls back to showing evidence without LLM analysis

### UX Improvement
Show clearer message to user when quota is exhausted:
```
⚠️ Free tier quota exceeded. Enter your OpenAI/Anthropic API key above for full analysis.
```

---

## Bug 6: Asyncio File Descriptor Warnings (Low Priority)

### Error
```
ValueError: Invalid file descriptor: -1
Exception ignored in: <function BaseEventLoop.__del__>
```

### Root Cause
Event loop cleanup issue in async code. Common when mixing sync/async or when event loops are garbage collected.

### Impact
**Cosmetic only** - doesn't affect functionality. Just pollutes logs.

### Fix (if desired)
Properly close event loops or use `asyncio.run()` context managers.

---

## Priority Order

1. **Bug 4 (HSDD)** - 2 min fix, improves UX immediately
2. **Bug 2 (Loading indicator)** - 5 min fix, critical for UX
3. **Bug 3 (Temperature)** - Needs investigation, breaks advanced mode
4. **Bug 1 (API key)** - Needs Gradio research, workaround exists (enter key after clicking example)
5. **Bug 5 (Quota message)** - Nice to have
6. **Bug 6 (Asyncio)** - Low priority, cosmetic

---

## Test Plan
- [ ] Fix HSDD acronym
- [ ] Add loading indicator yield
- [ ] Test advanced mode with temperature fix
- [ ] Research Gradio example behavior for API key
- [ ] Run `make check`
- [ ] Deploy and test on HuggingFace Spaces
