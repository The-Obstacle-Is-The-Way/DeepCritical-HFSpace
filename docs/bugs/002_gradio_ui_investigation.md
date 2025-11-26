# Bug Investigation: Gradio UI Header Cutoff in HuggingFace Spaces

**Date:** November 26, 2025
**Investigator:** Gemini (Supreme World Expert in Gradio/HF)
**Status:** Documented (Pending Implementation)
**Target Bug:** Top content (Title/Markdown) cut off/hidden under HF banner.

## 1. The Problem
The Gradio application deployed on HuggingFace Spaces displays a layout issue where the top-most content (Title and Description Markdown) is obscured by the HuggingFace Spaces header/banner. Users are unable to scroll up to reveal this content.

**Context:**
- **SDK:** Gradio `6.0.1`
- **Environment:** HuggingFace Spaces (Docker/Gradio SDK)
- **Configuration:** `header: mini` in `README.md`
- **Code Structure:** `gr.Blocks(fill_height=True)` containing `gr.Markdown` followed by `gr.ChatInterface`.

## 2. Root Cause Analysis
The issue stems from a conflict between **Gradio's `fill_height=True` layout engine**, the **HuggingFace Spaces iframe environment**, and the **placement of content outside `ChatInterface`**.

1.  **`fill_height=True` Behavior:** When `fill_height=True` is set on `gr.Blocks`, Gradio applies CSS to force the container to take up 100% of the viewport height (`100vh`) and uses a flex column layout.
2.  **Iframe & Banner Conflict:** In HuggingFace Spaces, the app runs inside an iframe. The "Mini Header" (`header: mini`) or the standard header floats over or pushes the iframe content. When Gradio forces `100vh`, it calculates based on the *window* or *iframe* size. If the top padding isn't handled correctly by the browser's flex calculation in this context, the top element (Markdown) gets pushed up or obscured because the flex container tries to fit the massive `ChatInterface` (which also wants to fill height) into the view.
3.  **Component Structure:** `ChatInterface` is designed to be a full-page component. Placing `gr.Markdown` *above* it while `fill_height=True` is active on the *parent* creates a layout competition. The parent tries to fit both, but `ChatInterface` consumes all available space, potentially causing overflow issues at the top rather than the bottom, or messing up the scroll anchor.

## 3. Investigation Findings

### GitHub & Web Search
- **Similar Issues:** Multiple reports exist of "header cutoff" in Spaces when using custom layouts or `fill_height`.
- **CSS Workarounds:** Common fixes involve manually adding `margin-top` or `padding-top` to `.gradio-container` or `body`.
- **Gradio 5/6 Changes:** Gradio 5.x introduced a more aggressive `fill_height` system. While it fixes many internal scrolling issues, it assumes it owns the entire viewport, which is only partially true in an embedded Space with a header.

### Code Analysis (`src/app.py`)
```python
with gr.Blocks(
    title="DeepCritical...",
    fill_height=True,  # <--- THE CULPRIT
) as demo:
    gr.Markdown(...)   # <--- HIDDEN CONTENT
    gr.ChatInterface(...)
```
The `fill_height=True` is applied to the *entire* app, forcing the Markdown + ChatInterface to squeeze into the viewport.

## 4. Potential Solutions

### Solution A: Structural Fix (Recommended)
Move the title and description *inside* the `ChatInterface` component. `ChatInterface` natively supports `title` and `description` parameters. This allows the component to handle the layout and scrolling of the header internally, ensuring it respects the `fill_height` logic correctly.

**Why:** This is the "Gradio-native" way. It prevents layout fighting between the Markdown block and the Chat block.

**Code Change:**
```python
gr.ChatInterface(
    fn=research_agent,
    title="🧬 DeepCritical",
    description="## AI-Powered Drug Repurposing Research Agent\n\nAsk questions about...",
    # ... other params
)
# Remove the separate gr.Markdown
```

### Solution B: CSS Workaround (Brittle)
Force a top margin to clear the header.

**Why:** Quick fix, but depends on the exact height of the HF header (which can change).

**Code Change:**
```css
.gradio-container {
    margin-top: 40px !important; /* Adjust based on header size */
}
```

### Solution C: Remove `fill_height` (Safe Fallback)
Remove `fill_height=True` from `gr.Blocks`.

**Why:** This returns to standard document flow. The page will scroll normally. The downside is the chat window might not be "sticky" at the bottom of the screen, requiring full page scrolling.

## 5. Recommended Action Plan

We will proceed with **Solution A (Structural Fix)** as it is the most robust and architecturally correct solution.

1.  **Modify `src/app.py`**:
    -   Extract the Markdown content.
    -   Pass it into `gr.ChatInterface(title=..., description=...)`.
    -   Remove the standalone `gr.Markdown` component.
    -   Keep `fill_height=True` (or let ChatInterface handle it default) to ensure the chat stays full-screen but with the header properly integrated.

2.  **Alternative**: If Solution A is not desired (e.g., complex markdown needed that `description` doesn't support well), we will apply **Solution B (CSS)** with `padding-top: 50px`.

## 6. Next Steps
Await approval to apply Solution A to `src/app.py`.
