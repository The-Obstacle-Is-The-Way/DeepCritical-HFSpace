# Bug 003: UI Viewport Overflow - Redesign Required

**Date:** November 27, 2025
**Status:** Active
**Severity:** High (Demo UX)
**Related:** 001_gradio_ui_cutoff.md, 002_gradio_ui_investigation.md

## Problem Summary

The Gradio UI has too much content to fit in a single viewport when embedded in HuggingFace Spaces iframe. The header gets cut off on initial load due to iframe rendering behavior.

**Key observation:** When user zooms out (cmd-), the iframe re-renders and everything becomes visible. This confirms it's a viewport/content overflow issue, not a pure CSS bug.

## Current UI Elements (Top to Bottom)

```
1. Title: "DeepCritical"
2. Subtitle: "AI-Powered Drug Repurposing Research Agent"
3. Description paragraph
4. Example questions (3 bullet points)
5. Orchestrator Mode selector (simple/magentic)
6. API Key input (BYOK)
7. API Provider selector (openai/anthropic)
8. Chatbot component (large)
9. Examples table (4 rows with Mode, API Key, Provider columns)
10. MCP Tools section header
11. MCP Tools tabs (PubMed, Clinical Trials, Preprints, Search All, Analyze)
12. Selected tool interface (Query input, Max Results slider, buttons)
13. Footer disclaimer
```

**Problem:** This is ~13 distinct UI sections. Too much for one screen.

## What Changed Recently

| Component | Before | After | Purpose |
|-----------|--------|-------|---------|
| Orchestrator Mode | Not visible | simple/magentic toggle | Multi-agent support |
| API Key | Not visible | Text input (password) | BYOK for paid tiers |
| API Provider | Not visible | openai/anthropic toggle | Support both providers |
| Backend info | None | "Free Tier (HF Inference)" banner | User knows what's running |

These additions (BYOK, provider selection, mode selection) added ~100px+ of vertical space.

## Root Cause Analysis

1. **Iframe height constraint:** HF Spaces iframe has fixed viewport
2. **Content overflow:** 13 sections exceed viewport height
3. **Gradio ChatInterface:** Designed to fill available space, but when there's content above AND below, it can't calculate correctly
4. **No scroll on initial load:** Iframe rendering doesn't trigger scroll to top

## Redesign Options

### Option A: Minimal Header (Recommended)

Remove redundant/verbose content. Keep only essentials above the chatbot.

```
BEFORE (verbose):
┌─────────────────────────────────────────┐
│ 🧬 DeepCritical                         │
│ AI-Powered Drug Repurposing Research... │
│ Ask questions about potential drug...   │
│ Example questions:                      │
│ • "What drugs could be repurposed..."   │
│ • "Is metformin effective..."           │
│ • "What existing medications..."        │
├─────────────────────────────────────────┤
│ Orchestrator Mode: [simple] [magentic]  │
│ API Key: [____________]                 │
│ API Provider: [openai] [anthropic]      │
├─────────────────────────────────────────┤
│ Chatbot                                 │
│                                         │
└─────────────────────────────────────────┘

AFTER (minimal):
┌─────────────────────────────────────────┐
│ 🧬 DeepCritical - Drug Repurposing AI   │
├─────────────────────────────────────────┤
│ Chatbot                                 │
│                                         │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│ ⚙️ Settings (accordion - collapsed)     │
└─────────────────────────────────────────┘
```

**Changes:**
- One-line title (no subtitle/description)
- Examples ONLY in the chatbot examples feature (already there)
- Config in collapsed accordion (Gradio default for additional_inputs)
- Remove MCP Tools section entirely OR move to separate tab

### Option B: Tabbed Interface

```
┌─────────────────────────────────────────┐
│ 🧬 DeepCritical                         │
│ [Chat] [MCP Tools] [About]              │
├─────────────────────────────────────────┤
│                                         │
│ (Selected tab content fills space)      │
│                                         │
└─────────────────────────────────────────┘
```

**Changes:**
- Title only
- Tabs for different features
- Chat tab: Just chatbot + collapsed config
- MCP Tools tab: The individual search interfaces
- About tab: Description, examples, links

### Option C: Remove MCP Tools Section

The MCP Tools section (PubMed Search, Clinical Trials, etc.) duplicates functionality available through:
1. The main chatbot (searches automatically)
2. Claude Desktop via MCP protocol

**Rationale:** For a hackathon demo, users care about the CHAT experience. Power users can use MCP.

```
┌─────────────────────────────────────────┐
│ 🧬 DeepCritical - Drug Repurposing AI   │
│ Ask about drug repurposing opportunities│
├─────────────────────────────────────────┤
│ Chatbot                                 │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│ ⚙️ Settings ▼                           │
│ (collapsed: mode, api key, provider)    │
├─────────────────────────────────────────┤
│ MCP: /gradio_api/mcp/ for Claude Desktop│
└─────────────────────────────────────────┘
```

## Recommendation

**Option A + C combined:**

1. **Minimal header:** One line title + one line description
2. **Remove example bullets:** Keep only in ChatInterface examples feature
3. **Keep config in accordion:** Already using `additional_inputs` (collapsed by default)
4. **Remove MCP Tools UI:** Keep MCP server active, remove the tab interfaces
5. **Minimal footer:** One line with MCP endpoint info

**Estimated vertical savings:** ~400px (should fit in single viewport)

## Implementation Checklist

- [ ] Simplify gr.Markdown header to 2 lines max
- [ ] Remove MCP Tools gr.Tab section from UI
- [ ] Verify additional_inputs accordion is collapsed by default
- [ ] Add single-line footer with MCP endpoint
- [ ] Test in HF Spaces iframe at 100% zoom
- [ ] Test on mobile viewport

## Files to Modify

```
src/app.py  - Main UI layout changes
```

## Success Criteria

- [ ] Full UI visible without scrolling at 100% zoom
- [ ] Header ("DeepCritical") always visible
- [ ] Chatbot functional
- [ ] Config accessible (in accordion)
- [ ] MCP still works (server active, just no UI tabs)
