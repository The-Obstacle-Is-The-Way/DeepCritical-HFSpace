# P2 Bug: First Agent Turn Exceeds Workflow Timeout

**Date**: 2025-12-03
**Status**: FIXED (PR fix/p2-double-bug-squash)
**Severity**: P2 (UX - Workflow always times out on complex queries)
**Component**: `src/orchestrators/advanced.py` + `src/agents/search_agent.py`
**Affects**: Both Free Tier (HuggingFace) AND Paid Tier (OpenAI)

---

## Executive Summary

The search agent's first turn can exceed the 5-minute workflow timeout, causing:
1. `iterations=0` at timeout (no agent completed a turn)
2. `_handle_timeout()` synthesizes from partial evidence
3. Users get incomplete research results

This is a **performance/architecture bug**, not a model issue.

---

## Symptom

```
[warning] Workflow timed out             iterations=0
```

The workflow times out with `iterations=0` - meaning the first agent (search agent) never completed its turn before the 5-minute timeout.

---

## Root Cause

The search agent's first turn is **extremely expensive**:

```
Search Agent First Turn:
├── Manager assigns task
├── Search agent starts
│   ├── Calls PubMed search tool (10 results)
│   ├── Calls ClinicalTrials search tool (10 results)
│   ├── Calls EuropePMC search tool (10 results)
│   └── For EACH result (30 total):
│       ├── Generate embedding (OpenAI API call)
│       ├── Check for duplicates (ChromaDB query)
│       └── Store in ChromaDB
│
│   TOTAL: 30 results × (embedding + dedup + store) = 90+ API/DB operations
│
└── Agent turn completes (if timeout hasn't fired)
```

**The timeout is on the WORKFLOW, not individual agent turns.** A single greedy agent can consume the entire timeout budget.

---

## Impact

| Aspect | Impact |
|--------|--------|
| UX | Queries always timeout on first turn |
| Research quality | Synthesis happens on partial evidence |
| Confusion | `iterations=0` looks like nothing happened |

---

## The Fix (Consensus)

**Reduce work per turn + increase timeout budget.**

### Implementation

**1. Reduce results per tool (immediate)**

`src/agents/search_agent.py` line 70:
```python
# Change from 10 to 5
result: SearchResult = await self._handler.execute(query, max_results_per_tool=5)
```

**2. Increase workflow timeout (immediate)**

`src/utils/config.py`:
```python
advanced_timeout: float = Field(
    default=600.0,  # Was 300.0 (5 min), now 10 min
    ge=60.0,
    le=900.0,
    description="Timeout for Advanced mode in seconds",
)
```

### Why NOT Per-Turn Timeout

**DANGER**: The SearchHandler uses `asyncio.gather()`:

```python
# src/tools/search_handler.py line 163-164
results = await asyncio.gather(*tasks, return_exceptions=True)
```

This is an **all-or-nothing** operation. If you wrap it with `asyncio.timeout()` and the timeout fires, you get **zero results**, not partial results.

```python
# DON'T DO THIS - yields nothing on timeout
async with asyncio.timeout(60):
    result = await self._handler.execute(query)  # Cancelled = zero results
```

Per-turn timeout requires `SearchHandler` to support cancellation with partial results. That's a separate architectural change (see Future Work).

---

## Future Work (Streaming Evidence Ingestion)

For proper fix, `SearchHandler.execute()` should:
1. Yield results as they arrive (async generator)
2. Support cancellation with partial results
3. Allow agent to return "what we have so far" on timeout

```python
# Future architecture
async def execute_streaming(self, query: str) -> AsyncIterator[Evidence]:
    for tool in self.tools:
        async for evidence in tool.search_streaming(query):
            yield evidence  # Can be cancelled at any point
```

This is out of scope for the immediate fix.

---

## Test Plan

1. Run query with 10-minute timeout
2. Verify first agent turn completes before timeout
3. Verify `iterations >= 1` at workflow end

---

## Verification Data

From diagnostic run:
```
=== RAW FRAMEWORK EVENTS ===
  MagenticAgentDeltaEvent: 284
  MagenticOrchestratorMessageEvent: 3
  ...
  NO MagenticAgentMessageEvent  ← Agent never completed a turn!

[warning] Workflow timed out             iterations=0
```

---

## Related

- P2 Duplicate Report Bug (separate issue, happens after successful completion)
- `_handle_timeout()` correctly synthesizes, but with partial evidence
- Not related to model quality - this is infrastructure/performance
