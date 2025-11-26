# BUG-001: Gradio UI Header Cut Off on HuggingFace Spaces

**Status:** 🟡 Fix Implemented (Pending Deploy)
**Severity:** Medium (UI/UX)
**Reported:** 2025-11-26
**Branch:** `fix/gradio-ui-scrolling`
**Related GitHub Issue:** #31

---

## Problem Description

The top portion of the Gradio app (title, description, markdown blocks) is cut off and hidden beneath the HuggingFace Spaces banner. Users cannot scroll up to see this content.

**Symptoms:**
- Top content (title "🧬 DeepCritical", description) invisible
- Unable to scroll up to see hidden content
- Issue disappears when resizing browser or zooming in/out
- Most prominent on mobile devices
- Works fine when running locally (without HF banner)

---

## Root Cause Analysis

### Finding 1: SDK Version Mismatch

| Location | Gradio Version |
|----------|----------------|
| Local development (`uv pip show gradio`) | **6.0.1** |
| HuggingFace Spaces (`README.md` frontmatter) | **5.33.0** |
| Bug fix released in | **5.36** |

**Root Cause:** HuggingFace Spaces is deploying with `sdk_version: "5.33.0"` which is **before** the fix in 5.36.

### Finding 2: Known Gradio Issue

This is a **known issue** reported during the same MCP hackathon:

- **GitHub Issue:** [#11417 - Top of gradio app cut off by huggingface banner](https://github.com/gradio-app/gradio/issues/11417)
- **Status:** Closed (Fixed)
- **Fixed in:** Gradio 5.36 via [PR #11427](https://github.com/gradio-app/gradio/pull/11427)
- **Fix description:** "Rendering of visible components" improvement

### Finding 3: Missing Layout Parameters

Current `app.py` does **not** use recommended layout parameters:

```python
# Current (problematic):
with gr.Blocks(title="DeepCritical - Drug Repurposing Research Agent") as demo:

# Recommended:
with gr.Blocks(title="...", fill_height=True) as demo:
```

---

## Solution

### Primary Fix: Upgrade SDK Version

Update `README.md` frontmatter:

```yaml
# Before (broken):
sdk_version: "5.33.0"

# After (fixed - use 5.36+ or latest 6.x):
sdk_version: "6.0.1"
```

### Secondary Fix: Add fill_height Parameter

Update `src/app.py`:

```python
with gr.Blocks(
    title="DeepCritical - Drug Repurposing Research Agent",
    fill_height=True,  # <-- Add this
) as demo:
```

### Tertiary Fix: CSS Workaround (if needed)

```python
css = """
#chatbot {
    flex-grow: 1 !important;
    overflow: auto !important;
}
"""

with gr.Blocks(
    title="...",
    fill_height=True,
    css=css,
) as demo:
```

---

## References

- [GitHub Issue #11417](https://github.com/gradio-app/gradio/issues/11417) - Original bug report (same hackathon!)
- [GitHub PR #11427](https://github.com/gradio-app/gradio/pull/11427) - Fix PR
- [Stack Overflow: Gradio ChatInterface CSS height](https://stackoverflow.com/questions/79084620/how-can-i-adjust-the-height-of-a-gradio-chatinterface-component-using-css)
- [Stack Overflow: Gradio screen appearance](https://stackoverflow.com/questions/78536986/a-chat-with-gradio-how-to-modify-its-screen-appearance)
- [GitHub Issue #9923](https://github.com/gradio-app/gradio/issues/9923) - ChatInterface squished in Blocks

---

## Action Items

1. [x] Update `README.md` sdk_version from "5.33.0" to "6.0.1" (or latest) - **DONE**
2. [x] Add `fill_height=True` to `gr.Blocks()` in `src/app.py` - **DONE**
3. [x] Run `make check` - 101 tests passed
4. [ ] Deploy to HuggingFace Spaces and verify fix
5. [ ] Close GitHub Issue #31 when verified

---

## Investigation Timeline

| Time | Action | Finding |
|------|--------|---------|
| 18:52 | Created branch `fix/gradio-ui-scrolling` | - |
| 18:52 | Web searched "Gradio 2025 ChatInterface top content cut off" | Found Issue #11417 |
| 18:53 | Fetched Issue #11417 details | Fixed in 5.36 via PR #11427 |
| 18:53 | Checked local Gradio version | 6.0.1 |
| 18:54 | Checked README.md sdk_version | **5.33.0** (before fix!) |
| 18:55 | Confirmed root cause | SDK version mismatch |
| 19:00 | Updated README.md sdk_version | "5.33.0" → "6.0.1" |
| 19:00 | Added fill_height=True to app.py | Both fixes applied |
| 19:02 | Ran `make check` | 101 tests passed |
