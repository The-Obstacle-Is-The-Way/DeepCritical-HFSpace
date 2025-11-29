# UI/UX Brainstorm: Mode Selection & API Key Experience

**Date**: 2025-11-28
**Status**: IMPLEMENTED (2025-11-28)
**Related**: Issues #52, #53, PR #58

---

## CRITICAL FINDING: Anthropic Key is Nearly Useless

**Code verification** (2025-11-28):
```
grep -r "AnthropicChatClient" src/  → NO RESULTS
grep -r "OpenAIChatClient" src/     → 22 RESULTS (all Magentic agents)
```

The `agent-framework` package (Microsoft's Magentic) **ONLY** has `OpenAIChatClient`.
There is no `AnthropicChatClient`. This means:

| Feature | OpenAI Key | Anthropic Key |
|---------|------------|---------------|
| Simple mode (Judge LLM) | ✅ GPT-5.1 | ✅ Claude Sonnet 4.5 |
| Advanced mode (Multi-agent) | ✅ Full orchestration | ❌ **DOES NOT WORK** |
| Value proposition | Full access | Simple mode only |

**Decision**: Keep Anthropic support for Simple mode, but ensure UX clearly differentiates capabilities.

---

## Current State (After PR #58)

### What Users See (Screenshot 2025-11-28)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ≡ Examples                                                                   │
├──────────────────────────────────────────────────────┬──────────────────────┤
│                                                      │ Orchestrator Mode    │
├──────────────────────────────────────────────────────┼──────────────────────┤
│ What drugs improve female libido post-menopause?     │ simple               │
│ Clinical trials for erectile dysfunction altern...   │ advanced             │
│ Evidence for testosterone therapy in women with...   │ simple               │
└──────────────────────────────────────────────────────┴──────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚙️ Mode & API Key (Free tier works!)                                  [▼]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Orchestrator Mode                                                           │
│ ⚡ Simple: Fast (Free/Any Key) | 🔬 Advanced: Deep Multi-Agent (OpenAI Key Only)    │
│ [● simple] [○ advanced]                                                     │
│                                                                             │
│ 🔑 API Key (Optional)                                                       │
│ Leave empty for free tier. Auto-detects provider from key prefix.           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ sk-... (OpenAI) or sk-ant-... (Anthropic)                               │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Observations from Screenshot

1. **Examples table**: 2 columns (Query + Mode) - clean, one example now shows "advanced" ✅
2. **One example shows "advanced"**: Improves discoverability of Advanced mode ✅
3. **Accordion collapsed by default**: Still collapsed, but with more inviting label ✅
4. **Placeholder mentions Anthropic**: Correct, but now clearly tied to Simple mode only via info text ✅
5. **"Advanced: Requires OpenAI key"**: Now more prominent with emojis and clearer phrasing in info text ✅

### The Two Modes

| Mode | Backend | Capabilities | Requirements |
|------|---------|--------------|--------------|
| **Simple** | Linear orchestrator | Search → Judge → Report (single pass) | None (free tier) or any API key |
| **Advanced** | Magentic multi-agent | SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent working together with iterative refinement | **OpenAI API key only** |

---

## Problems Identified (Addressed)

### P1: Advanced Mode is Hidden → ADDRESSED
- **Fix**: One example now shows "advanced" mode.
- **Fix**: Accordion label is more descriptive.

### P2: Mode/Key Relationship is Unclear → ADDRESSED
- **Fix**: `gr.Radio` info text clearly states "OpenAI Key Only" for Advanced mode, using emojis for emphasis.

### P3: No Incentive to Try Advanced → PARTIALLY ADDRESSED
- **Fix**: Emojis and "Deep Multi-Agent" hint at the value. Further marketing/documentation still needed for full "wow" moment.

### P4: Anthropic Users Left Out → ADDRESSED (Clarified)
- **Fix**: Anthropic keys still work for Simple mode, and the info text clarifies the limitation for Advanced mode.

---

## Options to Consider (Decision Made)

The recommendation of **Modified Option A (Better Education + Examples)** with slight modification to accordion label was implemented.

---

## Implementation Notes (Completed)

```python
# From src/app.py
examples=[
    ["What drugs improve female libido post-menopause?", "simple"],
    ["Clinical trials for erectile dysfunction alternatives to PDE5 inhibitors?", "advanced"],  # Changed
    ["Evidence for testosterone therapy in women with HSDD?", "simple"],
],

additional_inputs_accordion=gr.Accordion(
    label="⚙️ Mode & API Key (Free tier works!)", # Changed
    open=False
),

gr.Radio(
    choices=["simple", "advanced"],
    value="simple",
    label="Orchestrator Mode",
    info=( # Changed
        "⚡ Simple: Fast (Free/Any Key) | "
        "🔬 Advanced: Deep Multi-Agent (OpenAI Key Only)"
    ),
),
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-28 | Implemented Modified Option A | Minimal changes, high impact on discoverability, graceful fallback, user-approved accordion label. |