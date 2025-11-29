# P1 Bug: Gradio Settings Accordion Not Collapsing

**Priority**: P1 (UX Bug)
**Status**: OPEN
**Date**: 2025-11-27
**Target Component**: `src/app.py`

---

## 1. Problem Description

The "Settings" accordion in the Gradio UI (containing Orchestrator Mode, API Key, Provider) fails to collapse, even when configured with `open=False`. It remains permanently expanded, cluttering the interface and obscuring the chat history.

### Symptoms
- Accordion arrow toggles visually, but content remains visible.
- Occurs in both local development (`uv run src/app.py`) and HuggingFace Spaces.

---

## 2. Root Cause Analysis

**Definitive Cause**: Nested `Blocks` Context Bug.
`gr.ChatInterface` is itself a high-level abstraction that creates a `gr.Blocks` context. Wrapping `gr.ChatInterface` inside an external `with gr.Blocks():` context causes event listener conflicts, specifically breaking the JavaScript state management for `additional_inputs_accordion`.

**Reference**: [Gradio Issue #8861](https://github.com/gradio-app/gradio/issues/8861) confirms that `additional_inputs_accordion` malfunctions when `ChatInterface` is not the top-level block.

---

## 3. Solution Strategy: "The Unwrap Fix"

We will remove the redundant `gr.Blocks` wrapper. This restores the native behavior of `ChatInterface`, ensuring the accordion respects `open=False`.

### Implementation Plan

**Refactor `src/app.py` / `create_demo()`**:

1.  **Remove** the `with gr.Blocks() as demo:` context manager.
2.  **Instantiate** `gr.ChatInterface` directly as the `demo` object.
3.  **Migrate UI Elements**:
    *   **Header**: Move the H1/Title text into the `title` parameter of `ChatInterface`.
    *   **Footer**: Move the footer text ("MCP Server Active...") into the `description` parameter. `ChatInterface` supports Markdown in `description`, making it the ideal place for static info below the title but above the chat.

### Before (Buggy)
```python
def create_demo():
    with gr.Blocks() as demo:  # <--- CAUSE OF BUG
        gr.Markdown("# Title")
        gr.ChatInterface(..., additional_inputs_accordion=gr.Accordion(open=False))
        gr.Markdown("Footer")
    return demo
```

### After (Correct)
```python
def create_demo():
    return gr.ChatInterface(   # <--- FIX: Top-level component
        ...,
        title="🧬 DeepBoner",
        description="*AI-Powered Drug Repurposing Agent...*\n\n---\n**MCP Server Active**...",
        additional_inputs_accordion=gr.Accordion(label="⚙️ Settings", open=False)
    )
```

---

## 4. Validation

1.  **Run**: `uv run python src/app.py`
2.  **Check**: Open `http://localhost:7860`
3.  **Verify**:
    *   Settings accordion starts **COLLAPSED**.
    *   Header title ("DeepBoner") is visible.
    *   Footer text ("MCP Server Active") is visible in the description area.
    *   Chat functionality works (Magentic/Simple modes).

---

## 5. Constraints & Notes

- **Layout**: We lose the ability to place arbitrary elements *below* the chat box (footer will move to top, under title), but this is an acceptable trade-off for a working UI.
- **CSS**: `ChatInterface` handles its own CSS; any custom class styling from the previous footer will be standardized to the description text style.