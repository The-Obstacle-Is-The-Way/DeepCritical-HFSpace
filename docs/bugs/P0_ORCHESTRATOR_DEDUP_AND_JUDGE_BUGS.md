# P0 Bug Report: Orchestrator Dedup + Judge Failures

## Status
- **Date:** 2025-11-29
- **Priority:** P0 (Blocker - Simple mode broken on HF Spaces)
- **Component:** `src/orchestrator.py`, `src/agent_factory/judges.py`
- **Resolution:** FIXED in commits `5e761eb`, `2588375`

---

## Symptoms

When running Simple mode (free tier) on HuggingFace Spaces:

1. **Judge always returns 0% confidence** → loops forever with "continue"
2. **Deduplication removes ALL evidence** after iteration 1
3. **Never synthesizes** → user sees infinite loop

### Example Output

```
📚 SEARCH_COMPLETE: Found 20 new sources (19 total)   ← Iteration 1 OK
✅ JUDGE_COMPLETE: Assessment: continue (confidence: 0%)  ← FAIL: 0% = fallback

📚 SEARCH_COMPLETE: Found 12 new sources (11 total)   ← Iteration 2 BROKEN
...
📚 SEARCH_COMPLETE: Found 31 new sources (0 total)    ← 0 TOTAL = all removed!
✅ JUDGE_COMPLETE: Assessment: continue (confidence: 0%)  ← Still failing
```

---

## Root Cause Analysis

### Bug 1: Semantic Deduplication Removes Old Evidence

**File:** `src/orchestrator.py:213-219`

```python
# URL dedup (correct)
seen_urls = {e.citation.url for e in all_evidence}
unique_new = [e for e in new_evidence if e.citation.url not in seen_urls]
all_evidence.extend(unique_new)

# BUG: Passes ALL evidence (including old) to semantic dedup
all_evidence = await self._deduplicate_and_rank(all_evidence, query)
```

**Problem:** The `deduplicate()` function checks each item against the vector store. Items from iteration 1 are ALREADY in the store. When re-checked in iteration 2+, they find THEMSELVES (distance ≈ 0) and are removed as "duplicates".

**Result:** After iteration 1, evidence count drops to 0.

### Bug 2: HF Inference Judge Always Failing

**File:** `src/agent_factory/judges.py:186-254`

**Evidence:** Judge returns this every time:
- `confidence: 0.0`
- `recommendation: "continue"`
- Next queries are just the original query with suffixes

This is the `_create_fallback_assessment()` response, meaning:
- The HF Inference API calls are failing
- All 3 fallback models (Llama, Mistral, Zephyr) are failing
- Likely due to rate limits, quota, or model availability

---

## The Fix

### Fix 1: Only Dedup NEW Evidence (not all_evidence)

```python
# Before (broken)
all_evidence.extend(unique_new)
all_evidence = await self._deduplicate_and_rank(all_evidence, query)

# After (fixed)
# Only dedup the NEW evidence against the store
if unique_new:
    unique_new = await self._deduplicate_new_evidence(unique_new, query)
all_evidence.extend(unique_new)
```

Or simpler - disable semantic dedup until we fix it properly:

```python
# Disable broken semantic dedup
# all_evidence = await self._deduplicate_and_rank(all_evidence, query)
```

### Fix 2: Handle HF Inference Failures Gracefully

Option A: After N failed judge calls, force synthesize with available evidence
Option B: Increase retry count or add longer backoff
Option C: Fall back to MockJudgeHandler (which DOES work) after failures

```python
# In _create_fallback_assessment, track failures
if self._consecutive_failures >= 3:
    # Force synthesis instead of infinite loop
    return JudgeAssessment(
        sufficient=True,  # STOP
        confidence=0.1,
        recommendation="synthesize",
        ...
    )
```

---

## Test Plan

- [ ] Disable semantic dedup OR fix to only process new items
- [ ] Verify evidence accumulates across iterations (not drops to 0)
- [ ] Test HF Inference with fresh HF_TOKEN
- [ ] If HF keeps failing, fall back to MockJudgeHandler
- [ ] Verify "synthesize" is eventually reached
- [ ] Deploy and test on HF Space

---

## Priority Justification

**P0** because:
- Simple mode (free tier) is the DEFAULT experience
- Currently produces infinite loop with no output
- Users see "confidence: 0%" and think tool is broken
- Blocks hackathon demo for users without API keys

---

## Quick Workaround

Disable semantic dedup by setting `enable_embeddings=False` in orchestrator creation:

```python
orchestrator = create_orchestrator(
    ...
    enable_embeddings=False,  # Disable broken dedup
)
```

Or users can enter an OpenAI/Anthropic API key to bypass HF Inference issues.
