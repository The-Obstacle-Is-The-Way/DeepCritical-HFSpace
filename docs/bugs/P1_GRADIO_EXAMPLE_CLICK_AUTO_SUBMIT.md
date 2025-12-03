# P1: Gradio Example Click Auto-Submits Instead of Loading

**Status:** OPEN
**Priority:** P1 (High - UX breaks BYOK flow)
**Discovered:** 2025-12-03
**Component:** `src/app.py` (Gradio UI)

---

## Summary

Clicking on example questions in the Gradio ChatInterface immediately starts the research agent instead of just loading the text into the input field. This prevents users from:
1. Entering their API key before starting the chat
2. Modifying the example query before submission
3. Understanding what's happening (chat starts without explicit action)

---

## Reproduction Steps

1. Open DeepBoner Gradio UI
2. **Before entering any API key**, click on an example like "What drugs improve female libido post-menopause?"
3. Observe: Chat immediately starts with Free Tier
4. Try to enter an OpenAI API key in the accordion
5. Try to submit a new query
6. **Result:** Confusing UX - the chat already ran, state is unclear

### Expected Behavior

1. Click example → text loads into input field
2. User can enter API key
3. User clicks submit → chat starts with their configured settings

---

## Root Cause Analysis

### Problem 1: Missing `run_examples_on_click=False`

Gradio's `ChatInterface` has a parameter `run_examples_on_click` (added in [PR #10109](https://github.com/gradio-app/gradio/pull/10109), December 2024):

| Value | Behavior |
|-------|----------|
| `True` (default) | Clicking example immediately runs the function |
| `False` | Clicking example only populates the input field |

**Our code** in `src/app.py:279-325` does NOT set this parameter:

```python
demo = gr.ChatInterface(
    fn=research_agent,
    examples=[...],
    # run_examples_on_click=False  ← MISSING!
)
```

### Problem 2: HuggingFace Spaces Default Overrides

From [Gradio docs](https://www.gradio.app/docs/gradio/chatinterface):

> `cache_examples`: The default option in HuggingFace Spaces is **True**.
> `run_examples_on_click` has **no effect** if `cache_examples` is True.

This means on HuggingFace Spaces:
- `cache_examples` defaults to `True`
- Even if we add `run_examples_on_click=False`, it would be **ignored**
- We MUST explicitly set `cache_examples=False`

### ~~Problem 3: Example Data Overwrites User Settings~~ (CORRECTION: This is Actually Fine)

Looking at lines 283-304:

```python
examples=[
    [
        "What drugs improve female libido post-menopause?",
        "sexual_health",
        None,  # ← api_key set to None
        None,  # ← api_key_state set to None
    ],
    ...
]
```

**CORRECTION:** Per [Stack Overflow research](https://stackoverflow.com/questions/78584977/how-to-use-additional-inputs-and-examples-at-the-same-time):

> "If you set None for some input in all examples then it will not display this column in example and example will not change current value for this input."

Since ALL examples have `None` for api_key and api_key_state:
- Those columns won't display in the examples table
- **Clicking an example will NOT change the API key textbox**
- User's API key is PRESERVED!

The current example structure is actually **correct**. The only issue is auto-submit.

### Dead Code: api_key_state Never Updated (Non-Blocking)

Line 258-259 has a comment suggesting a fix was attempted:

```python
# BUG FIX: Add gr.State for API key persistence across example clicks
api_key_state = gr.State("")
```

This code is **dead** because:
1. The `gr.State` is initialized empty (`""`)
2. There's NO event handler (`.change()`) to update the state when textbox changes
3. The value passed to `research_agent` is always `""`
4. In `_validate_inputs`: `(api_key or api_key_state or "")` - the State never contributes

**However**, this is NOT blocking the fix. The fix works regardless of this dead code.
We can clean it up in a separate PR after the fix is verified working.

---

## Architecture Implications

### BYOK Flow Broken

The unified architecture (SPEC-16) relies on API key auto-detection:

```
User provides key?
├── YES → OpenAI backend (sk-...) or Anthropic backend (sk-ant-...)
└── NO  → HuggingFace Free Tier
```

The example click bug forces users into Free Tier even if they intended to use their API key.

### Session State Confusion

After an example auto-submits:
1. Chat history has content
2. User enters API key
3. User submits new query
4. **Question:** Does the new query use the new key? Is history preserved correctly?

This creates ambiguous state that could lead to:
- Inconsistent backend usage within a session
- Confusion about which tier was used for which response

---

## Fix Implementation

### Required Changes to `src/app.py`

```python
demo = gr.ChatInterface(
    fn=research_agent,
    title="🍆 DeepBoner",
    description=description,
    examples=[...],
    additional_inputs_accordion=additional_inputs_accordion,
    additional_inputs=[...],
    # === FIX: Prevent auto-submit on example click ===
    cache_examples=False,  # MUST be False for run_examples_on_click to work
    run_examples_on_click=False,  # Load into input, don't auto-run
)
```

### Why This Fix is Safe (No Optional Enhancements Needed)

The current example structure with `None` values is **correct**:
- API key textbox value is PRESERVED when clicking examples
- Only the message textbox is populated
- No restructuring of examples needed

**The fix is minimal and surgical:**
```python
cache_examples=False,
run_examples_on_click=False,
```

No other changes required.

---

## Testing

### Manual Test Cases

1. **Fresh load, click example:** Should only populate input, not start chat
2. **Enter API key, click example:** Query loads, API key preserved
3. **Click example, enter key, submit:** Should use the entered key
4. **Multiple example clicks:** Each should just replace input text

### Automated Test (if possible)

```python
def test_example_click_does_not_auto_submit():
    """Verify examples only populate input, not trigger function."""
    # Would need Gradio testing utilities
    pass
```

---

## Related Issues

- [Gradio #10103](https://github.com/gradio-app/gradio/issues/10103): Original feature request for `run_examples_on_click`
- [Gradio #10109](https://github.com/gradio-app/gradio/pull/10109): PR that implemented the parameter
- SPEC-16: Unified Chat Client Architecture (relies on proper API key handling)
- P2_ARCHITECTURAL_BYOK_GAPS.md (archived) - Related BYOK issues now fixed

---

## Priority Justification

**P1 (High)** because:
1. Breaks the BYOK (Bring Your Own Key) user flow
2. Forces users into Free Tier unexpectedly
3. Creates confusing UX that may prevent demo adoption
4. Simple fix with clear solution path

---

## Files Affected

- `src/app.py:279-325` - ChatInterface configuration

---

## Senior Review: Risk Assessment

**Reviewed:** 2025-12-03

### Verification Performed

1. **Gradio Version Confirmed:** 6.0.1 (`uv pip show gradio`)
2. **Parameters Exist:** Both `run_examples_on_click` and `cache_examples` verified in `ChatInterface.__init__` signature
3. **No Hidden Gradio Usage:** Only `src/app.py` imports gradio (grep confirmed)
4. **No Event Handlers:** No `.change()`, `.click()`, `.submit()` events in app.py that could conflict
5. **Example Format Correct:** List-of-lists format matches `additional_inputs` order

### Potential Regressions Checked

| Risk | Assessment | Mitigation |
|------|------------|------------|
| Cold start slower on HF Spaces | Low - examples aren't pre-cached, but they also don't run on click | None needed - acceptable tradeoff |
| Progress bar issues | None - `gr.Progress()` issues only affect cached examples, we're disabling caching | N/A |
| Example display changes | None - examples already appear below chatbot due to `additional_inputs` | N/A |
| API key cleared on example click | **Verified SAFE** - `None` in all examples means input is preserved | N/A |
| Dead State code causes issues | No - it's inert, just passes `""` always | Clean up in follow-up PR |

### Gotchas Investigated

1. **ViewFrame/hydration issues:** `ssr_mode=False` already set at line 339 - no conflict
2. **MCP server interaction:** MCP server (`mcp_server=True`) operates independently of examples - no conflict
3. **CSS injection:** Custom CSS only affects `.api-key-input` class - no conflict
4. **Accordion state:** `additional_inputs_accordion` unaffected by example behavior

### Confidence Level

**HIGH** - This is a two-line, surgical fix that:
- Uses documented, stable Gradio 6.0 parameters
- Has no side effects on other components
- Preserves existing example structure
- Was explicitly designed for this use case (PR #10109)

### Recommended Approach

1. **Phase 1:** Add the two params, test manually on HF Spaces
2. **Phase 2:** (Optional) Clean up dead `api_key_state` code in follow-up PR

---

## References

- [Gradio ChatInterface Docs](https://www.gradio.app/docs/gradio/chatinterface)
- [Gradio Examples Behavior](https://www.gradio.app/guides/chatinterface-examples)
- [PR #10109: run_examples_on_click](https://github.com/gradio-app/gradio/pull/10109)
- [Stack Overflow: None values in examples](https://stackoverflow.com/questions/78584977/how-to-use-additional-inputs-and-examples-at-the-same-time)
