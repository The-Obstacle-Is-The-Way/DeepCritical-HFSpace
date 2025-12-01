# P3 Bug: Progress Bar Positioning/Overlap in ChatInterface

**Severity**: P3 (Low - UX polish)
**Status**: Open
**Discovered**: 2025-12-01
**Reporter**: Internal QA

## Symptom

The `gr.Progress()` bar renders in a strange position when used inside `ChatInterface`:
- Progress bar appears to "float" in the middle of the chat output
- Text overlaps with progress bar elements
- Bar appears static/stuck at certain percentages
- Visual artifact: `Round 2/5 (~2m 15s remaining) - 48.0%` renders both above and inside bar

## Screenshot Evidence

Progress bar appears inline between chat messages rather than in a fixed position (top/bottom of component).

## Root Cause Analysis

### The Conflict

We're mixing **two different progress mechanisms**:

1. **`gr.Progress()`** - Gradio's general-purpose progress bar API
   - Designed for `gr.Interface` and `gr.Blocks` functions
   - Renders as an overlay on the output component

2. **`ChatInterface.show_progress`** - Built-in chat progress
   - Options: `"full"`, `"minimal"`, `"hidden"`
   - `"full"` = spinner + runtime display
   - `"minimal"` = runtime display only (default)

When both are used together in `ChatInterface`, the `gr.Progress()` bar fights for position with the chat's streaming output, causing visual glitches.

### Current Implementation (`src/app.py`)

```python
async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    ...
    progress: gr.Progress = gr.Progress(),  # <- ISSUE: Manual progress bar
) -> AsyncGenerator[str, None]:
    ...
    if event.type == "started":
        progress(0, desc="Starting research...")  # <- Updates overlay
    elif event.type == "progress":
        progress(p, desc=event.message)  # <- Conflicts with chat streaming
```

### Gradio Documentation

From [Gradio ChatInterface Docs](https://www.gradio.app/docs/gradio/chatinterface):

> `show_progress`: how to show the progress animation while event is running: 'full' shows a spinner which covers the output component area as well as a runtime display in the upper right corner, 'minimal' only shows the runtime display, 'hidden' shows no progress animation at all

From [GitHub Issue #5967](https://github.com/gradio-app/gradio/issues/5967):

> `gr.Progress` is "not integrated with ChatInterface or Chatbots" - known limitation.

## Impact

| Aspect | Impact |
|--------|--------|
| Functionality | None - app works correctly |
| UX | Visual confusion, looks unprofessional |
| Accessibility | May cause screen reader confusion |

## Proposed Solutions

### Option 1: Remove `gr.Progress()` - Use Chat Text Only (RECOMMENDED)

Remove the `gr.Progress()` parameter entirely and rely on our existing emoji status messages:

```python
async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    domain: str = "sexual_health",
    api_key: str = "",
    api_key_state: str = "",
    # REMOVED: progress: gr.Progress = gr.Progress(),
) -> AsyncGenerator[str, None]:
    ...
    # Keep emoji status updates in chat output
    # ⏱️ **PROGRESS**: Round 1/5 (~3m 0s remaining)
    # These are already being yielded to the chat
```

**Pros**:
- Simplest solution
- No CSS hacks
- Status is visible in chat history
- Works with ChatInterface's built-in progress

**Cons**:
- No visual progress bar (just text status)

### Option 2: Use `show_progress="full"` Parameter

Add explicit `show_progress` to ChatInterface and remove `gr.Progress()`:

```python
demo = gr.ChatInterface(
    fn=research_agent,
    title="🍆 DeepBoner",
    show_progress="full",  # Built-in spinner + runtime
    ...
)
```

**Pros**:
- Uses Gradio's intended mechanism
- Consistent with Gradio UX patterns

**Cons**:
- Less granular (no percentage, no custom desc)

### Option 3: Custom Progress Component with CSS (COMPLEX)

Wrap ChatInterface in `gr.Blocks` and add a separate `gr.HTML` progress bar outside the chat:

```python
with gr.Blocks() as demo:
    progress_html = gr.HTML("<div id='custom-progress'></div>", visible=False)
    chat = gr.ChatInterface(...)

    # Update progress_html separately from chat
```

**Pros**:
- Full control over positioning
- Can match exact design requirements

**Cons**:
- Significant refactor required
- May break MCP server integration
- More moving parts = more bugs

### Option 4: Hybrid - Progress in `additional_inputs_accordion`

Place a separate `gr.Progress` component in a fixed position (e.g., accordion header):

```python
with gr.Blocks() as demo:
    with gr.Row():
        progress_bar = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
    chat = gr.ChatInterface(...)
```

**Pros**:
- Fixed position, no overlap

**Cons**:
- Requires `gr.Blocks` wrapper (breaks MCP?)
- Clunky UX

## Recommended Fix

**Option 1: Remove `gr.Progress()`, keep emoji status text**

Rationale:
1. Our emoji status updates (`⏱️ **PROGRESS**: Round 2/5`) already provide the information
2. `gr.Progress()` was never designed for ChatInterface
3. Removing it eliminates the positioning conflict entirely
4. ChatInterface's built-in `show_progress="minimal"` handles the spinner

## Implementation Plan

1. **Remove** `progress: gr.Progress = gr.Progress()` parameter from `research_agent()`
2. **Remove** all `progress(...)` calls in the function
3. **Keep** emoji status yields (`⏱️ **PROGRESS**: ...`)
4. **Optionally** add `show_progress="minimal"` to ChatInterface (already default)
5. **Test** on HuggingFace Spaces

## Testing

```bash
# Local test
uv run python -c "from src.app import create_demo; demo, _ = create_demo(); demo.launch()"

# Verify:
# 1. No floating progress bar
# 2. Emoji status updates visible in chat
# 3. ChatInterface spinner works as expected
```

## References

- [Gradio Progress Bars Guide](https://www.gradio.app/guides/progress-bars)
- [Gradio ChatInterface Docs](https://www.gradio.app/docs/gradio/chatinterface)
- [GitHub Issue #5967: Progress bar for ChatInterface](https://github.com/gradio-app/gradio/issues/5967)
- [GitHub Issue #5815: Customize the progress bar](https://github.com/gradio-app/gradio/issues/5815)

## Version Info

- Gradio: `>=6.0.0` (pyproject.toml constraint)
- ChatInterface: Streaming async generator mode
- MCP Server: Enabled
