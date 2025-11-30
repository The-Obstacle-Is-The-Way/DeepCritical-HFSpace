# SPEC 01: Demo Termination & Timing Fix

## Priority: P0 (Hackathon Blocker)

## Problem Statement

Advanced (Magentic) mode runs indefinitely from user perspective. The demo was manually terminated after ~10 minutes without reaching synthesis.

**Root Cause Hypothesis**: We're trusting `agent_framework.MagenticBuilder.max_round_count` to enforce termination, but:
1. We don't know how the framework counts "rounds"
2. Our `iteration` counter only tracks `MagenticAgentMessageEvent`, not all framework rounds
3. Manager coordination messages (JUDGING) happen between rounds and don't count

## Investigation Required

### Question 1: Does max_round_count actually work?

```python
# Current code (src/orchestrator_magentic.py:111)
.with_standard_manager(
    chat_client=manager_client,
    max_round_count=self._max_rounds,  # Default: 10
    max_stall_count=3,
    max_reset_count=2,
)
```

**Test**: Set `max_round_count=2` and verify termination.

### Question 2: What counts as a "round"?

From demo output:
- `JUDGING` (Manager) - many of these
- `SEARCH_COMPLETE` (Agent)
- `HYPOTHESIZING` (Agent)
- `JUDGE_COMPLETE` (Agent)
- `STREAMING` (Delta events)

Is one "round" = one full cycle of all agents? Or one agent message?

### Question 3: Why no final synthesis?

The demo showed lots of evidence gathering but never reached `ReportAgent`. Either:
1. JudgeAgent never said "sufficient=True"
2. Framework terminated before synthesis (unlikely given time)
3. Something else broke the flow

## Proposed Solutions

### Option A: Add Hard Timeout (Recommended for Hackathon)

```python
# src/orchestrator_magentic.py
import asyncio

async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
    # ...existing setup...

    DEMO_TIMEOUT_SECONDS = 300  # 5 minutes max

    try:
        async with asyncio.timeout(DEMO_TIMEOUT_SECONDS):
            async for event in workflow.run_stream(task):
                # ...existing processing...

    except TimeoutError:
        yield AgentEvent(
            type="complete",
            message="Research timed out. Synthesizing available evidence...",
            data={"reason": "timeout", "iterations": iteration},
            iteration=iteration,
        )
        # Attempt to synthesize whatever we have
```

### Option B: Reduce max_rounds AND Add Progress

```python
# Lower the round count AND show which round we're on
max_round_count=5,  # Was 10
```

Plus yield round number:
```python
yield AgentEvent(
    type="progress",
    message=f"Round {round_num}/{max_rounds}...",
    iteration=round_num,
)
```

### Option C: Force Synthesis After N Evidence Items

```python
# In judge logic
if len(evidence) >= 20:
    return "synthesize"  # We have enough, stop searching
```

## Acceptance Criteria

- [x] Demo completes in <5 minutes with visible progress
- [x] User sees round count (e.g., "Round 3/5")
- [x] Always produces SOME output (even if partial)
- [x] Timeout prevents infinite running

**Status: IMPLEMENTED** (commit b1d094d)

## Test Plan

```python
@pytest.mark.asyncio
async def test_magentic_terminates_within_timeout():
    """Verify demo completes in reasonable time."""
    orchestrator = MagenticOrchestrator(max_rounds=3)

    events = []
    start = time.time()

    async for event in orchestrator.run("simple test query"):
        events.append(event)
        if time.time() - start > 120:  # 2 min max for test
            pytest.fail("Orchestrator did not terminate")

    # Must have a completion event
    assert any(e.type == "complete" for e in events)
```

## Related Issues

- #65: P1: Advanced Mode takes too long for hackathon demo
- #47: E2E Testing

## Files to Modify

1. `src/orchestrator_magentic.py` - Add timeout and progress
2. `src/app.py` - Display round progress in UI
3. `tests/unit/test_magentic_termination.py` - Add timeout test
