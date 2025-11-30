# P3 Bug Report: Advanced Mode Missing Termination Guarantee

## Status
- **Date:** 2025-11-29
- **Priority:** P3 (Edge case, but confusing UX)
- **Component:** `src/orchestrator_magentic.py`
- **Resolution:** Fixed (Guarantee termination event)

---

## Symptoms

In **Advanced (Magentic) mode** with OpenAI API key:

1. Workflow runs for many iterations (up to 10 rounds)
2. Agents search, judge, hypothesize repeatedly
3. Eventually... **nothing happens**
   - No "complete" event
   - No error message
   - UI just stops updating

**User perception:** "Did it finish? Did it crash? What happened?"

### Observed Behavior

When workflow hits `max_round_count=10`:
- `workflow.run_stream(task)` iterator ends
- NO `MagenticFinalResultEvent` is emitted by agent-framework
- Our code yields nothing after the loop
- User is left hanging

---

## Root Cause Analysis

### Code Path (`src/orchestrator_magentic.py:170-186`)

```python
iteration = 0
try:
    async for event in workflow.run_stream(task):
        agent_event = self._process_event(event, iteration)
        if agent_event:
            if isinstance(event, MagenticAgentMessageEvent):
                iteration += 1
            yield agent_event
    # BUG: NO FALLBACK HERE!
    # If loop ends without FinalResultEvent, user sees nothing

except Exception as e:
    logger.error("Magentic workflow failed", error=str(e))
    yield AgentEvent(
        type="error",
        message=f"Workflow error: {e!s}",
        iteration=iteration,
    )
# BUG: NO FINALLY BLOCK TO GUARANTEE TERMINATION EVENT
```

### Workflow Configuration (`src/orchestrator_magentic.py:110-116`)

```python
.with_standard_manager(
    chat_client=manager_client,
    max_round_count=self._max_rounds,  # 10 - can hit this limit
    max_stall_count=3,                  # If agents repeat 3x
    max_reset_count=2,                  # Workflow reset limit
)
```

### Failure Modes

| Scenario | What Happens | User Sees |
|----------|--------------|-----------|
| `MagenticFinalResultEvent` emitted | `_process_event` yields "complete" | Final report |
| Max rounds (10) reached, no final event | Loop ends silently | **Nothing** |
| `max_stall_count` triggered | Workflow ends | **Nothing** |
| `max_reset_count` triggered | Workflow ends | **Nothing** |
| OpenAI API error | Exception caught | Error message |

---

## The Fix

Add guaranteed termination event after the loop:

```python
iteration = 0
final_event_received = False

try:
    async for event in workflow.run_stream(task):
        agent_event = self._process_event(event, iteration)
        if agent_event:
            if isinstance(event, MagenticAgentMessageEvent):
                iteration += 1
            if agent_event.type == "complete":
                final_event_received = True
            yield agent_event

except Exception as e:
    logger.error("Magentic workflow failed", error=str(e))
    yield AgentEvent(
        type="error",
        message=f"Workflow error: {e!s}",
        iteration=iteration,
    )
    final_event_received = True  # Error is a form of termination

finally:
    # GUARANTEE: Always emit termination event
    if not final_event_received:
        logger.warning(
            "Workflow ended without final event",
            iterations=iteration,
        )
        yield AgentEvent(
            type="complete",
            message=(
                f"Research completed after {iteration} agent rounds. "
                "Max iterations reached - results may be partial. "
                "Try a more specific query for better results."
            ),
            data={"iterations": iteration, "reason": "max_rounds_reached"},
            iteration=iteration,
        )
```

---

## Alternative: Increase Max Rounds

The default `max_rounds=10` might be too low for complex queries.

In `src/orchestrator_factory.py:52-53`:
```python
return orchestrator_cls(
    max_rounds=config.max_iterations if config else 10,  # Could increase to 15-20
    api_key=api_key,
)
```

**Trade-off:** More rounds = more API cost, but better chance of complete results.

---

## Test Plan

- [ ] Add fallback yield after async for loop
- [ ] Add `final_event_received` flag tracking
- [ ] Log warning when fallback is used
- [ ] Test with `max_rounds=2` to force hitting limit
- [ ] Verify user always sees termination event
- [ ] `make check` passes

---

## Related Files

- `src/orchestrator_magentic.py` - Main fix location
- `src/orchestrator_factory.py` - Max rounds configuration
- `src/utils/models.py` - AgentEvent types
- `docs/bugs/P2_MAGENTIC_THINKING_STATE.md` - Related UX issue (implemented)

---

## Priority Justification

**P3** because:
- Advanced mode is working for most queries
- Only hits edge case when max rounds reached without synthesis
- User CAN retry with different query
- Not blocking hackathon demo (free tier Simple mode works)

Would be P2 if:
- This happened frequently
- No workaround existed
