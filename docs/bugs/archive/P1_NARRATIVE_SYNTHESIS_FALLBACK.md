# P1: Narrative Synthesis Falls Back to Template (SPEC_12 Not Taking Effect)

**Status**: Open
**Priority**: P1 - Major UX degradation
**Affects**: Simple mode, all deployments
**Root Cause**: LLM synthesis silently failing → template fallback
**Related**: SPEC_12 (implemented but not functioning)

---

## Problem Statement

SPEC_12 implemented LLM-based narrative synthesis, but users still see **template-formatted bullet points** instead of **prose paragraphs**:

### What Users See (Template Fallback)

```markdown
## Sexual Health Analysis

### Question
what medication for the best boners?

### Drug Candidates
- **tadalafil**
- **sildenafil**

### Key Findings
- Tadalafil improves erectile function

### Assessment
- **Mechanism Score**: 4/10
- **Clinical Evidence Score**: 6/10
```

### What They Should See (LLM Synthesis)

```markdown
### Executive Summary

Sildenafil demonstrates clinically meaningful efficacy for erectile dysfunction,
with strong evidence from multiple RCTs demonstrating improved erectile function...

### Background

Erectile dysfunction (ED) is a common male sexual health disorder...

### Evidence Synthesis

**Mechanism of Action**
Sildenafil works by inhibiting phosphodiesterase type 5 (PDE5)...
```

---

## Root Cause Analysis

### Location: `src/orchestrators/simple.py:555-564`

```python
try:
    agent = Agent(model=get_model(), output_type=str, system_prompt=system_prompt)
    result = await agent.run(user_prompt)
    narrative = result.output
except Exception as e:  # ← SILENT FALLBACK
    logger.warning("LLM synthesis failed, using template fallback", error=str(e))
    return self._generate_template_synthesis(query, evidence, assessment)
```

**The Problem**: When ANY exception occurs during LLM synthesis, it silently falls back to template. Users see janky bullet points with no indication that the LLM call failed.

### Why Synthesis Fails

| Cause | Symptom | Frequency |
|-------|---------|-----------|
| No API key in deployment | HuggingFace Spaces | HIGH |
| API rate limiting | Heavy usage | MEDIUM |
| Token overflow | Long evidence lists | MEDIUM |
| Model mismatch | Wrong model ID | LOW |
| Network timeout | Slow connections | LOW |

---

## Evidence: LLM Synthesis WORKS When Configured

Local test with API key:
```python
# This works perfectly:
agent = Agent(model=get_model(), output_type=str, system_prompt=system_prompt)
result = await agent.run(user_prompt)
print(result.output)  # → Beautiful narrative prose!
```

Output:
```
### Executive Summary

Sildenafil demonstrates clinically meaningful efficacy for erectile dysfunction,
with one study (Smith, 2020; N=100) reporting improved erectile function...
```

---

## Impact

| Metric | Current | Expected |
|--------|---------|----------|
| Report quality | 3/10 (metadata dump) | 9/10 (professional prose) |
| User satisfaction | Low | High |
| Clinical utility | Limited | High |

The ENTIRE VALUE PROPOSITION of the research agent is the synthesized report. Template output defeats the purpose.

---

## Fix Options

### Option A: Surface Error to User (RECOMMENDED)

When LLM synthesis fails, don't silently fall back. Show the user what went wrong:

```python
except Exception as e:
    logger.error("LLM synthesis failed", error=str(e), exc_info=True)

    # Show error in report instead of silent fallback
    error_note = f"""
⚠️ **Note**: AI narrative synthesis unavailable.
Showing structured summary instead.

_Technical: {type(e).__name__}: {str(e)[:100]}_
"""
    template = self._generate_template_synthesis(query, evidence, assessment)
    return f"{error_note}\n\n{template}"
```

### Option B: HuggingFace Secrets Configuration

For HuggingFace Spaces deployment, add secrets:
- `OPENAI_API_KEY` → Required for synthesis
- `ANTHROPIC_API_KEY` → Alternative provider

### Option C: Graceful Degradation with Explanation

Add a banner explaining synthesis status:
- ✅ "AI-synthesized narrative report" (when LLM works)
- ⚠️ "Structured summary (AI synthesis unavailable)" (fallback)

---

## Diagnostic Steps

To determine why synthesis is failing in production:

1. **Review logs** for warning: `"LLM synthesis failed, using template fallback"`
2. **Verify API key**: Is `OPENAI_API_KEY` set in environment?
3. **Confirm model access**: Is `gpt-5` accessible with current API tier?
4. **Inspect rate limits**: Is the account quota exhausted?

---

## Acceptance Criteria

- [ ] Users see narrative prose reports (not bullet points) when API key is configured
- [ ] When synthesis fails, user sees clear indication (not silent fallback)
- [ ] HuggingFace Spaces deployment has proper secrets configured
- [ ] Logging captures the specific exception for debugging

---

## Files to Modify

| File | Change |
|------|--------|
| `src/orchestrators/simple.py:555-580` | Add error surfacing in fallback |
| `src/app.py` | Add synthesis status indicator to UI |
| HuggingFace Spaces Settings | Add `OPENAI_API_KEY` secret |

---

## Test Plan

1. Run locally with API key → Should get narrative prose
2. Run locally WITHOUT API key → Should get template WITH error message
3. Deploy to HuggingFace with secrets → Should get narrative prose
4. Deploy to HuggingFace WITHOUT secrets → Should get template WITH warning
