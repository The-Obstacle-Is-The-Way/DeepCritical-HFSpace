# P1 Bug: No Synthesis Report in Free Tier (Premature Workflow Termination)

**Date**: 2025-12-04
**Status**: FIXED (PR fix/p1-forced-synthesis)
**Severity**: P1 (Critical UX - No usable output from research)
**Component**: `src/orchestrators/advanced.py`
**Affects**: Free Tier (HuggingFace) primarily, potentially Paid Tier

---

## Executive Summary

The workflow terminates without the ReportAgent ever producing a synthesis report. Users see search results and hypotheses streaming, but the final output is just "Research complete." with no actual research report. This is caused by the 7B Manager model failing to properly delegate to ReportAgent before workflow termination.

---

## Symptom

```text
📚 **SEARCH_COMPLETE**: searcher: [search results]
⏱️ **PROGRESS**: Round 1/5 (~3m 0s remaining)
🔬 **HYPOTHESIZING**: hypothesizer: [hypotheses]
⏱️ **PROGRESS**: Round 2/5 (~2m 15s remaining)
✅ **JUDGE_COMPLETE**: judge: [asks for more evidence]
⏱️ **PROGRESS**: Round 4/5 (~45s remaining)
Research complete.
Research complete.   ← NO SYNTHESIS REPORT!
```

The workflow runs through multiple agents (Search, Hypothesis, Judge) but never reaches the ReportAgent. The user receives no usable research report.

---

## Root Cause Analysis

### Primary Issue: Manager Model Failure

The `with_standard_manager()` in Microsoft Agent Framework uses the provided chat client (HuggingFace 7B model) to coordinate agents. The 7B model:

1. **Cannot follow complex multi-step instructions** - The manager prompt instructs: "When JudgeAgent says SUFFICIENT EVIDENCE → delegate to ReportAgent." The 7B model doesn't reliably follow this.

2. **Triggers premature termination** - The framework has `max_stall_count=3` and `max_reset_count=2`. If the manager keeps making the same delegation or gets confused, the workflow terminates.

3. **Emits final event without synthesis** - The framework sends `MagenticFinalResultEvent` or `WorkflowOutputEvent` without ReportAgent ever running.

### Secondary Issue: Duplicate Complete Events

Both `MagenticFinalResultEvent` and `WorkflowOutputEvent` are emitted when the workflow ends. The previous code handled both, yielding "Research complete." twice.

---

## The Fix

### 1. Track ReportAgent Execution (Forced Synthesis)

Add a `reporter_ran` flag that tracks whether ReportAgent produced output:

```python
reporter_ran = False  # P1 FIX: Track if ReportAgent produced output

# In MagenticAgentMessageEvent handler:
agent_name = (event.agent_id or "").lower()
if "report" in agent_name:
    reporter_ran = True
```

### 2. Force Synthesis on Final Event

If the workflow ends without ReportAgent running, force synthesis:

```python
if isinstance(event, (MagenticFinalResultEvent, WorkflowOutputEvent)):
    if not reporter_ran:
        logger.warning("ReportAgent never ran - forcing synthesis")
        async for synth_event in self._force_synthesis(iteration):
            yield synth_event
    else:
        yield self._handle_final_event(event, iteration, last_streamed_length)
```

### 3. `_force_synthesis()` Method

Similar to `_handle_timeout()`, invokes ReportAgent directly:

```python
async def _force_synthesis(self, iteration: int) -> AsyncGenerator[AgentEvent, None]:
    """Force synthesis when workflow ends without ReportAgent running."""
    state = get_magentic_state()
    evidence_summary = await state.memory.get_context_summary()
    report_agent = create_report_agent(self._chat_client, domain=self.domain)

    yield AgentEvent(type="synthesizing", message="Synthesizing research findings...")

    synthesis_result = await report_agent.run(
        f"Synthesize research report from this evidence.\n\n{evidence_summary}"
    )

    yield AgentEvent(type="complete", message=synthesis_result.text)
```

### 4. Skip Duplicate Final Events

Prevent "Research complete." appearing twice:

```python
if isinstance(event, (MagenticFinalResultEvent, WorkflowOutputEvent)):
    if final_event_received:
        continue  # Skip duplicate final events
    final_event_received = True
```

---

## Why This Is The Correct Architecture

| Alternative | Why Wrong |
|-------------|-----------|
| Improve manager prompt | 7B models have fundamental reasoning limitations |
| Use larger model for manager | Defeats "free tier" purpose |
| Wait for upstream fix | Framework may never change; we control our code |
| **Forced synthesis safety net** | ✅ Guarantees output regardless of manager behavior |

The `_force_synthesis()` pattern is a **defensive architecture**. It guarantees users always get a research report, even if:
- The manager model fails to delegate properly
- The workflow hits stall/reset limits
- Any unexpected termination occurs

---

## Files Modified

| File | Change |
|------|--------|
| `src/orchestrators/advanced.py` | Added `reporter_ran` tracking |
| `src/orchestrators/advanced.py` | Added `_force_synthesis()` method |
| `src/orchestrators/advanced.py` | Added duplicate final event skipping |
| `src/orchestrators/advanced.py` | Added forced synthesis in final event handler |
| `src/orchestrators/advanced.py` | Added forced synthesis in max rounds fallback |

---

## Test Plan

1. **Free Tier**: Run query, verify synthesis report is always generated
2. **Paid Tier**: Run query, verify no regression in OpenAI behavior
3. **Timeout**: Verify existing timeout synthesis still works
4. **Max Rounds**: Verify synthesis happens even at max rounds

---

## Related

- P2 Duplicate Report Bug (separate issue, also fixed in this PR)
- P2 First Turn Timeout Bug (previously fixed)
- Manager model limitations are fundamental to 7B models
- OpenAI tier works because GPT-5 follows instructions better

---

## Lessons Learned

1. **Defensive architecture** - Don't trust upstream components to always behave correctly
2. **Tracking flags** - Simple boolean flags can enable powerful safety nets
3. **AI-native challenges** - When using AI models as infrastructure components, build in fallbacks for model failures
4. **Regression prevention** - This bug was likely introduced when we unified the architecture; comprehensive test coverage is critical
