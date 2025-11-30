# P2 Bug Report: Third Example Not Filling Chat Box

## Status
- **Date:** 2025-11-29
- **Priority:** P2 (UX issue)
- **Component:** `src/app.py` - Gradio examples
- **Resolution:** FIXED in commit `2ea01fd`

---

## Symptoms

When clicking the third example in the Gradio UI:
- **Example 1** (female libido): ✅ Fills chat box correctly
- **Example 2** (ED alternatives): ✅ Fills chat box correctly
- **Example 3** (HSDD testosterone): ❌ Does NOT fill chat box

### User Experience
User clicks example → nothing happens → confusion

---

## Root Cause Hypothesis

The third example contains parentheses and an abbreviation:
```
"Testosterone therapy for HSDD (Hypoactive Sexual Desire Disorder)?"
```

Possible causes:
1. **Parentheses** - Gradio may have parsing issues with `(...)` in example text
2. **Text length** - When expanded, this is the longest example
3. **Special characters** - The combination of abbreviation + parenthetical may confuse Gradio's example caching

---

## The Fix

Simplify the example text - expand the abbreviation and remove parentheses:

```python
# Before (broken)
"Testosterone therapy for HSDD (Hypoactive Sexual Desire Disorder)?"

# After (fixed)
"Testosterone therapy for Hypoactive Sexual Desire Disorder?"
```

This:
1. Removes problematic parentheses
2. Makes the text more readable (no cut-off abbreviation)
3. Users don't need to know what HSDD stands for

---

## Test Plan

- [ ] Change example text in `src/app.py`
- [ ] Deploy to HuggingFace Space
- [ ] Verify all 3 examples fill chat box correctly
- [ ] `make check` passes

---

## Related

- Gradio ChatInterface example caching behavior
- Similar to P0 example caching crash (but different manifestation)
