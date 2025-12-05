# P2 Bug: Round Counter Semantic Mismatch

**Status**: ✅ FIXED
**Discovered**: 2025-12-05
**Fixed**: 2025-12-05
**Severity**: P2 (Display bug, confusing UX but not blocking)
**Component**: `src/orchestrators/advanced.py`
**Commit**: `40ca236c refactor(orchestrator): implement semantic progress tracking`

---

## Symptom

Progress display shows impossible values like "Round 11/5":

```text
⏱️ **PROGRESS**: Round 11/5 (~0s remaining)
```

This is confusing to users - how can we be on round 11 when max is 5?

---

## Root Cause Analysis

### The Semantic Mismatch

Two different concepts are being conflated:

| Concept | What It Means | Variable |
|---------|---------------|----------|
| **Workflow Round** | One orchestration cycle where manager delegates to agents | `self._max_rounds` (5) |
| **Agent Completion** | One agent finishes its task | `state.iteration` (incremented on each `ExecutorCompletedEvent`) |

### The Bug

```python
# Line 348: Increments on EVERY agent completion
if isinstance(event, ExecutorCompletedEvent):
    state.iteration += 1

# Line 467: Displays as if it's a workflow round
message=f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)"
```

### Why It Happens

In a multi-agent workflow with 4 agents (searcher, hypothesizer, judge, reporter):
- Each "round" involves the manager delegating to multiple agents
- Each agent completion fires an `ExecutorCompletedEvent`
- With 4+ agents, we see 4+ events per workflow round

**Math**: 5 workflow rounds × 4 agents = 20+ agent completions, displayed as "Round 20/5"

---

## Evidence From Logs

The session showed this progression:
```text
Round 1/5   - First agent completed
Round 2/5   - Second agent completed
Round 3/5   - Third agent completed
Round 4/5   - Fourth agent completed
Round 5/5   - Fifth agent completed (still in workflow round 1!)
Round 6/5   - Now exceeds max (workflow round 2 starting)
...
Round 11/5  - Multiple workflow rounds have passed
```

---

## Impact

1. **User Confusion**: "Round 11/5" makes no sense
2. **Time Estimation Wrong**: `rounds_remaining = max(5 - 11, 0) = 0` → always shows "~0s remaining"
3. **No Actual Bug in Logic**: The workflow still runs correctly, just the display is wrong

---

## Proposed Fixes

### Option A: Rename to "Agent Step" (Quick Fix)

Change the display to reflect what we're actually counting:

```python
# Before
message=f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)"

# After
message=f"Agent step {iteration} (Round limit: {self._max_rounds})"
```

**Pros**: Accurate, minimal code change
**Cons**: Still doesn't track actual workflow rounds

### Option B: Track Actual Workflow Rounds (Proper Fix)

Track workflow rounds separately from agent completions:

```python
@dataclass
class WorkflowState:
    iteration: int = 0           # Agent completions (for internal tracking)
    workflow_round: int = 0      # Actual orchestration rounds
    current_message_buffer: str = ""
    # ...

# Increment workflow_round when manager delegates (different event type)
# Display workflow_round in progress messages
```

**Pros**: Semantically correct, accurate time estimates
**Cons**: Requires understanding which event signals a new round

### Option C: Use Estimated Agent Count (Compromise)

Estimate agents per round and display accordingly:

```python
AGENTS_PER_ROUND = 4  # searcher, hypothesizer, judge, reporter
estimated_round = (iteration // AGENTS_PER_ROUND) + 1
message=f"Round ~{estimated_round}/{self._max_rounds}"
```

**Pros**: Roughly accurate, no API research needed
**Cons**: Estimation may be off if some agents are skipped

---

## Recommendation

**Short-term**: Apply Option A (rename to "Agent step") - fixes the confusion immediately

**Long-term**: Investigate Option B - determine which event signals a new workflow round in Microsoft Agent Framework

---

## Related Code

```python
# src/orchestrators/advanced.py

# Line 348: Where iteration is incremented
if isinstance(event, ExecutorCompletedEvent):
    state.iteration += 1

# Line 459-467: Where progress message is generated
rounds_remaining = max(self._max_rounds - iteration, 0)
est_seconds = rounds_remaining * 45
progress_event = AgentEvent(
    type="progress",
    message=f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)",
    iteration=iteration,
)
```

---

## Test Case

```python
def test_progress_display_never_exceeds_max_rounds():
    """Progress should show Round X/Y where X <= Y."""
    # Simulate 20 agent completions across 5 workflow rounds
    # Assert displayed round never exceeds max_rounds
    pass
```

---

## Additional Issues Found During Analysis

### Issue 2: Dead Code - Unused `_get_progress_message` Method

```python
# Line 196-205: Method is defined but NEVER called
def _get_progress_message(self, iteration: int) -> str:
    """Generate progress message with time estimation."""
    # ... logic duplicated in _handle_completion_event
```

The same logic is duplicated inline in `_handle_completion_event` (lines 458-469).

**Fix**: Either use the method or delete it.

### Issue 3: Hardcoded Constant

```python
# Line 87: Class constant defined
_EST_SECONDS_PER_ROUND: int = 45

# Line 199: Uses constant (correct)
est_seconds = rounds_remaining * self._EST_SECONDS_PER_ROUND

# Line 460: Uses hardcoded 45 (inconsistent)
est_seconds = rounds_remaining * 45
```

**Fix**: Use `self._EST_SECONDS_PER_ROUND` consistently.

### Issue 4: Time Estimate Always Shows "~0s remaining"

Since `iteration` quickly exceeds `max_rounds`:
```python
rounds_remaining = max(self._max_rounds - iteration, 0)
# When iteration=11, max_rounds=5: rounds_remaining = max(5-11, 0) = 0
# est_seconds = 0 * 45 = 0
# Display: "~0s remaining"
```

The time estimate becomes useless after the first few agent completions.

---

## Complete Fix Recommendation

1. **Rename display** from "Round X/5" to "Agent step X"
2. **Delete dead code** - remove unused `_get_progress_message` method
3. **Use constant** - replace hardcoded `45` with `self._EST_SECONDS_PER_ROUND`
4. **Fix time estimate** - base it on agent steps, not workflow rounds

---

## Senior Review Findings (2025-12-05)

**Reviewer**: External Gemini CLI Agent
**Status**: CONFIRMED - Analysis accurate and sufficient

### Additional Nuances Identified

1. **Manager Agent Also Fires Events**: The Manager itself is an agent. If `ExecutorCompletedEvent` fires for Manager's turn completion PLUS sub-agents' completions, the count accelerates 2-3x faster per logical round. This explains why we saw 11 events for ~2-3 workflow rounds.

2. **Time Estimation Doubly Flawed**:
   - Not just bottoming out at 0
   - `_EST_SECONDS_PER_ROUND` (45s) is calibrated for a FULL workflow round, not a single agent step
   - If we counted agent steps correctly: 10 steps × 45s = 450s (way overestimated)
   - A full round of 4 agents might only take 60s total

3. **API Discovery - Can Track Actual Rounds**:
   ```python
   # These constants exist in agent_framework:
   ORCH_MSG_KIND_INSTRUCTION = 'instruction'
   ORCH_MSG_KIND_USER_TASK = 'user_task'
   ORCH_MSG_KIND_TASK_LEDGER = 'task_ledger'
   ORCH_MSG_KIND_NOTICE = 'notice'
   ```

   Counting `user_task` events from `MagenticOrchestratorMessageEvent` would align iteration with `max_rounds` 1:1, since this signals "Manager is beginning a new evaluation cycle."

### Reviewer Recommendations

1. **Option A (Rename)**: APPROVED - Safest, most honest fix
2. **Option B (Track Workflow Rounds)**: DEFER - Requires verifying framework behavior across versions, risks brittleness
3. **Remove Denominator**: Display `Agent Step {iteration}` without `/5` to avoid confusion
4. **Delete Dead Code**: Confirmed `_get_progress_message` is never called
5. **Fix Constants**: Use `self._EST_SECONDS_PER_ROUND` consistently

### Review Status: ✅ PASSED - Ready for Implementation

---

## Resolution (2025-12-05)

**Implemented**: Domain-driven semantic progress tracking

### What Was Done

1. **Deleted Dead Code**:
   - Removed unused `_get_progress_message` method
   - Removed unused `_EST_SECONDS_PER_ROUND` constant

2. **Added Semantic Agent Mapping** (`_get_agent_semantic_name`):
   ```python
   def _get_agent_semantic_name(self, agent_id: str) -> str:
       """Map internal agent ID to user-facing semantic name."""
       name = agent_id.lower()
       if SEARCHER_AGENT_ID in name:
           return "SearchAgent"
       if JUDGE_AGENT_ID in name:
           return "JudgeAgent"
       if HYPOTHESIZER_AGENT_ID in name:
           return "HypothesisAgent"
       if REPORTER_AGENT_ID in name:
           return "ReportAgent"
       return "ManagerAgent"
   ```

3. **Changed Progress Display**:
   - Before: `"Round {iteration}/{self._max_rounds} (~{est_display} remaining)"`
   - After: `"Step {iteration}: {semantic_name} task completed"`

4. **Changed Initial Thinking Message**:
   - Before: `"Multi-agent reasoning in progress (5 rounds max)... Estimated time: 3-5 minutes."`
   - After: `"Multi-agent reasoning in progress (Limit: 5 Manager rounds)... Allocating time for deep research..."`

5. **Updated Tests**: Changed test mocks to use domain-specific agent IDs (`searcher`, `judge`) instead of arbitrary strings.

### Result

- Before: `⏱️ **PROGRESS**: Round 11/5 (~0s remaining)` (confusing, broken math)
- After: `⏱️ **PROGRESS**: Step 11: ReportAgent task completed` (accurate, professional)

### Design Decision

Rather than patching the counter display or trying to track "actual workflow rounds" (which requires deep framework integration), we chose **honest reporting**: Show exactly what happened (which agent completed) without making false promises about progress percentages or time estimates.

This follows the Clean Code principle: "Don't lie to the user."

---

## References

- SPEC-18: Agent Framework Core Upgrade (where ExecutorCompletedEvent was introduced)
- Microsoft Agent Framework documentation on workflow rounds vs agent executions
