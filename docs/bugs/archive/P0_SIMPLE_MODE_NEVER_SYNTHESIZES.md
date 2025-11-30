# P0 Bug Report: Simple Mode Never Synthesizes

## Status
- **Date:** 2025-11-29
- **Priority:** P0 (Blocker - Simple mode produces useless output)
- **Component:** `src/orchestrators/simple.py`, `src/agent_factory/judges.py`, `src/prompts/judge.py`
- **Environment:** Simple mode **WITHOUT OpenAI key** (HuggingFace Inference free tier)

---

## Symptoms

When running Simple mode with a real research question:

1. **Judge never recommends "synthesize"** even with 455 sources and 90% confidence
2. **Confidence drops to 0%** in late iterations (API failures or context overflow)
3. **Search derails** to tangential topics (bone health, muscle mass instead of libido)
4. **Max iterations reached** → User gets garbage output (just citations, no synthesis)

### Example Output (Real Run)

```
🔍 SEARCHING: What drugs improve female libido post-menopause?
📚 SEARCH_COMPLETE: Found 30 new sources (30 total)
✅ JUDGE_COMPLETE: Assessment: continue (confidence: 70%)    ← Never "synthesize"

... 8 more iterations ...

📚 SEARCH_COMPLETE: Found 10 new sources (429 total)
✅ JUDGE_COMPLETE: Assessment: continue (confidence: 0%)     ← API failure?

📚 SEARCH_COMPLETE: Found 26 new sources (455 total)
✅ JUDGE_COMPLETE: Assessment: continue (confidence: 0%)     ← Still failing

## Partial Analysis (Max Iterations Reached)      ← GARBAGE OUTPUT
### Question
What drugs improve female libido post-menopause?
### Status
Maximum search iterations reached.
### Citations
1. [Tribulus terrestris and female reproductive...]
2. ...
---
*Consider searching with more specific terms*     ← NO SYNTHESIS AT ALL
```

---

## Root Cause Analysis

### Bug 1: Judge Never Says "sufficient=True"

**File:** `src/prompts/judge.py:22-25`

```python
3. **Sufficiency**: Evidence is sufficient when:
   - Combined scores >= 12 AND
   - At least one specific drug candidate identified AND
   - Clear mechanistic rationale exists
```

**Problem:** The prompt is too conservative. With 455 sources spanning testosterone, DHEA, estrogen, oxytocin, etc., the judge should have identified candidates and said "synthesize". But:

1. LLM may not be extracting drug candidates from evidence properly
2. The "AND" conditions are too strict - evidence can be "good enough" without hitting all criteria
3. The recommendation "continue" seems to be the default state

**Evidence:** Output shows 70-90% confidence but still "continue" - the judge is confident but never satisfied.

### Bug 2: Confidence Drops to 0% (Late Iteration Failures)

**File:** `src/agent_factory/judges.py:150-183`

The `_create_fallback_assessment()` returns:
- `confidence: 0.0`
- `recommendation: "continue"`

**Problem:** In iterations 9-10, something failed:
- Context too long (455 sources × ~1500 chars = 680K chars → token limit exceeded)
- API rate limit hit
- Network timeout

**Evidence:** Confidence went from 80%→0%→0% in final iterations - this is the fallback response.

### Bug 3: Search Derailment

**Evidence from logs:**
```
Next searches: androgen therapy and bone health, androgen therapy and muscle mass...
Next searches: testosterone therapy in postmenopausal women, mechanisms of testosterone...
```

**Problem:** Judge's `next_search_queries` drift off-topic. "Bone health" and "muscle mass" are tangential to "female libido". The judge should stay focused on the original question.

### Bug 4: Partial Synthesis is Garbage

**File:** `src/orchestrators/simple.py:432-470`

```python
def _generate_partial_synthesis(self, query: str, evidence: list[Evidence]) -> str:
    """Generate a partial synthesis when max iterations reached."""
    citations = "\n".join([...])  # Just citations

    return f"""## Partial Analysis (Max Iterations Reached)
### Question
{query}
### Status
Maximum search iterations reached. The evidence gathered may be incomplete.
### Evidence Collected
Found {len(evidence)} sources.
### Citations
{citations}
---
*Consider searching with more specific terms*
"""
```

**Problem:** When max iterations reached, we have 455 sources but output NO analysis. We should:
1. Force a synthesis call to the LLM
2. Or at minimum generate drug candidates/findings from the last good assessment
3. Not just dump citations and give up

---

## The Fix

### Fix 1: Lower the Bar for "synthesize"

**Option A:** Change prompt to be less strict:
```python
SYSTEM_PROMPT = """...
3. **Sufficiency**: Evidence is sufficient when:
   - Combined scores >= 10 (was 12) OR
   - Confidence >= 80% with drug candidates identified OR
   - 5+ iterations completed with 100+ sources
"""
```

**Option B:** Add iteration-based heuristic in orchestrator:
```python
# If we have lots of evidence and high confidence, force synthesis
if iteration >= 5 and len(all_evidence) > 100 and assessment.confidence > 0.7:
    assessment.sufficient = True
    assessment.recommendation = "synthesize"
```

### Fix 2: Handle Context Overflow

**File:** `src/agent_factory/judges.py`

Before sending to LLM, cap evidence:
```python
async def assess(self, question: str, evidence: list[Evidence]) -> JudgeAssessment:
    # Cap at 50 most recent/relevant to avoid token overflow
    if len(evidence) > 50:
        evidence = evidence[:50]  # Or use embedding similarity to select best 50
```

### Fix 3: Keep Search Focused

**File:** `src/prompts/judge.py`

Add to prompt:
```python
SYSTEM_PROMPT = """...
## Search Query Rules

When suggesting next_search_queries:
- Stay focused on the ORIGINAL question
- Do NOT drift to tangential topics (e.g., don't search "bone health" for a libido question)
- Refine existing good terms, don't explore random associations
"""
```

### Fix 4: Generate Real Synthesis on Max Iterations

**File:** `src/orchestrators/simple.py`

```python
def _generate_partial_synthesis(self, query: str, evidence: list[Evidence]) -> str:
    """Generate a REAL synthesis when max iterations reached."""

    # Get the last assessment's data (if available)
    last_assessment = self.history[-1]["assessment"] if self.history else None

    drug_candidates = last_assessment.get("details", {}).get("drug_candidates", []) if last_assessment else []
    key_findings = last_assessment.get("details", {}).get("key_findings", []) if last_assessment else []

    drug_list = "\n".join([f"- **{d}**" for d in drug_candidates]) or "- See sources below for candidates"
    findings_list = "\n".join([f"- {f}" for f in key_findings[:5]]) or "- Review citations for findings"

    citations = "\n".join([
        f"{i + 1}. [{e.citation.title}]({e.citation.url}) ({e.citation.source.upper()})"
        for i, e in enumerate(evidence[:10])
    ])

    return f"""## Drug Repurposing Analysis (Partial)

### Question
{query}

### Status
⚠️ Maximum iterations reached. Analysis based on {len(evidence)} sources.

### Drug Candidates Identified
{drug_list}

### Key Findings
{findings_list}

### Top Citations ({len(evidence)} sources)
{citations}

---
*Analysis may be incomplete. Consider refining query or adding API key for better results.*
"""
```

---

## Test Plan

- [ ] Verify judge says "synthesize" within 5 iterations for good queries
- [ ] Test with 500+ sources to ensure no token overflow
- [ ] Verify search stays on-topic (no bone/muscle tangents for libido query)
- [ ] Verify partial synthesis shows drug candidates (not just citations)
- [ ] Test with MockJudgeHandler to confirm issue is in LLM behavior
- [ ] Add unit test: `test_judge_synthesizes_with_good_evidence`

---

## Priority Justification

**P0** because:
- Simple mode is the DEFAULT for users without API keys
- 455 sources found but ZERO useful output generated
- User waited 10 iterations just to get a citation dump
- Makes the tool look completely broken
- Blocks hackathon demo effectiveness

---

## Immediate Workaround

1. Use **Advanced mode** (requires OpenAI key) - it has its own synthesis logic
2. Or use **fewer iterations** (MAX_ITERATIONS=3) to hit partial synthesis faster
3. Or manually review the citations (they ARE relevant, just not synthesized)

---

## Related Issues

- `P0_ORCHESTRATOR_DEDUP_AND_JUDGE_BUGS.md` - Fixed dedup issue, but synthesis problem persists
- `ACTIVE_BUGS.md` - Update when this is resolved
