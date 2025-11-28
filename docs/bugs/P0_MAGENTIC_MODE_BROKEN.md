# P0 Bug: Magentic Mode Returns ChatMessage Object Instead of Report Text

**Status**: OPEN
**Priority**: P0 (Critical)
**Date**: 2025-11-27

---

## Actual Bug Found (Not What We Thought)

**The OpenAI key works fine.** The real bug is different:

### The Problem

When Magentic mode completes, the final report returns a `ChatMessage` object instead of the actual text:

```
FINAL REPORT:
<agent_framework._types.ChatMessage object at 0x11db70310>
```

### Evidence

Full test output shows:
1. Magentic orchestrator starts correctly
2. SearchAgent finds evidence
3. HypothesisAgent generates hypotheses
4. JudgeAgent evaluates
5. **BUT**: Final output is `ChatMessage` object, not text

### Root Cause

In `src/orchestrator_magentic.py` line 193:

```python
elif isinstance(event, MagenticFinalResultEvent):
    text = event.message.text if event.message else "No result"
```

The `event.message` is a `ChatMessage` object, and `.text` may not extract the content correctly, or the message structure changed in the agent-framework library.

---

## Secondary Issue: Max Rounds Reached

The orchestrator hits max rounds before producing a report:

```
[ERROR] Magentic Orchestrator: Max round count reached
```

This means the workflow times out before the ReportAgent synthesizes the final output.

---

## What Works

- OpenAI API key: **Works** (loaded from .env)
- SearchAgent: **Works** (finds evidence from PubMed, ClinicalTrials, Europe PMC)
- HypothesisAgent: **Works** (generates Drug -> Target -> Pathway chains)
- JudgeAgent: **Partial** (evaluates but sometimes loses context)

---

## Files to Fix

| File | Line | Issue |
|------|------|-------|
| `src/orchestrator_magentic.py` | 193 | `event.message.text` returns object, not string |
| `src/orchestrator_magentic.py` | 97-99 | `max_round_count=3` too low for full pipeline |

---

## Suggested Fix

```python
# In _process_event, line 192-199
elif isinstance(event, MagenticFinalResultEvent):
    # Handle ChatMessage object properly
    if event.message:
        if hasattr(event.message, 'content'):
            text = event.message.content
        elif hasattr(event.message, 'text'):
            text = event.message.text
        else:
            text = str(event.message)
    else:
        text = "No result"
```

And increase rounds:

```python
# In _build_workflow, line 97
max_round_count=self._max_rounds,  # Use configured value, default 10
```

---

## Test Command

```bash
set -a && source .env && set +a && uv run python examples/orchestrator_demo/run_magentic.py "metformin alzheimer"
```

---

## Simple Mode Works

For reference, simple mode produces full reports:

```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin alzheimer"
```

Output includes structured report with Drug Candidates, Key Findings, etc.
