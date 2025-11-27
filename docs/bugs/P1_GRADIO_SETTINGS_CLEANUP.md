# P1 Bug: Gradio Settings Accordion Not Collapsing

**Priority**: P1 (UX Bug)
**Status**: OPEN
**Date**: 2025-11-27

---

## Bug Description

The "Settings" accordion in the Gradio UI does not collapse/hide its content. Even when the accordion arrow shows "collapsed" state, all settings (Orchestrator Mode, API Key, API Provider) remain visible.

---

## Root Cause

**Known Gradio Bug**: `additional_inputs_accordion` does not work correctly when `ChatInterface` is used inside `gr.Blocks()`.

**GitHub Issue**: [gradio-app/gradio#8861](https://github.com/gradio-app/gradio/issues/8861)
> "Is there any subsequent plan to support gr.ChatInterface inheritance under gr.Block()? Currently using accordion is not working well."

**Our Code** (`src/app.py` lines 196-250):
```python
with gr.Blocks(...) as demo:  # <-- Using gr.Blocks wrapper
    gr.ChatInterface(
        ...
        additional_inputs_accordion=gr.Accordion(label="⚙️ Settings", open=False),
        additional_inputs=[...],
    )
```

The `additional_inputs_accordion` parameter is designed for standalone `ChatInterface`, but breaks when wrapped in `gr.Blocks()`.

---

## Evidence

- Accordion arrow toggles (visual feedback works)
- Content does NOT hide when collapsed
- Same behavior in local dev and HuggingFace Spaces

---

## Possible Fixes

### Option 1: Remove gr.Blocks Wrapper (Recommended)

If we don't need the header/footer markdown, use standalone `ChatInterface`:

```python
# Instead of gr.Blocks wrapper
demo = gr.ChatInterface(
    fn=research_agent,
    title="🧬 DeepCritical",
    description="AI-Powered Drug Repurposing Agent",
    additional_inputs_accordion=gr.Accordion(label="⚙️ Settings", open=False),
    additional_inputs=[...],
)
```

**Pros**: Accordion should work correctly
**Cons**: Less control over layout, no custom header/footer

### Option 2: Manual Accordion Outside ChatInterface

Move settings outside `ChatInterface` into a proper `gr.Accordion`:

```python
with gr.Blocks() as demo:
    gr.Markdown("# 🧬 DeepCritical")

    with gr.Accordion("⚙️ Settings", open=False):
        mode = gr.Radio(choices=["simple", "magentic"], value="simple", label="Mode")
        api_key = gr.Textbox(label="API Key", type="password")
        provider = gr.Radio(choices=["openai", "anthropic"], value="openai", label="Provider")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a research question")

    msg.submit(research_agent, [msg, chatbot, mode, api_key, provider], chatbot)
```

**Pros**: Full control, accordion works
**Cons**: More code, lose ChatInterface conveniences (examples, etc.)

### Option 3: Wait for Gradio Fix

Gradio added `.expand()` and `.collapse()` events in recent versions. Upgrading might help.

**Check current version**:
```bash
pip show gradio | grep Version
```

**Upgrade**:
```bash
pip install --upgrade gradio
```

---

## Recommendation

**Option 1** (Remove gr.Blocks) is cleanest if we can live without custom header/footer.

If header/footer needed, **Option 2** gives working accordion at cost of more code.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/app.py` | Restructure UI per chosen option |
| `pyproject.toml` | Possibly upgrade Gradio version |

---

## Test Plan

1. Run locally: `uv run python -m src.app`
2. Click Settings accordion to collapse
3. Verify content hides when collapsed
4. Verify content shows when expanded
5. Test on HuggingFace Spaces after deploy

---

## Sources

- [Gradio Issue #8861 - Accordion not working in Blocks](https://github.com/gradio-app/gradio/issues/8861)
- [Gradio ChatInterface Docs](https://www.gradio.app/docs/gradio/chatinterface)
- [Gradio Accordion Docs](https://www.gradio.app/docs/gradio/accordion)
