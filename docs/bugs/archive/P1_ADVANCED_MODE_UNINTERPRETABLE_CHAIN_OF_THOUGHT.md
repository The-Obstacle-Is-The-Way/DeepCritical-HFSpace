# P1: Advanced Mode Exposes Uninterpretable Chain-of-Thought Events

**Priority**: P1 (UX Degradation)
**Component**: `src/orchestrators/advanced.py`
**Status**: Resolved
**Issue**: [#106](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/106)
**PR**: [#107](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/pull/107)
**Created**: 2025-12-01
**Resolved**: 2025-12-01

## Summary

The Advanced orchestrator exposes raw internal framework events from `agent-framework-core` directly to users. These events contain internal manager bookkeeping (task assignments, ledgers, instructions) that are:

1. Truncated mid-sentence at 200 characters
2. Use internal framework terminology (`user_task`, `task_ledger`, `instruction`)
3. Shown with misleading "JUDGING" event type
4. Not meaningful to end users

## Resolution

Implemented "Smart Filter + Transform" logic in `src/orchestrators/advanced.py`:

1. **Filtered**: `task_ledger` and `instruction` events are now hidden.
2. **Transformed**: `user_task` events are mapped to `type="progress"` with a friendly "Manager assigning research task..." message.
3. **Smart Truncation**: Text is now truncated at sentence boundaries or word boundaries, preventing mid-word cuts.

Verified with new unit tests in `tests/unit/orchestrators/test_advanced_events.py`.

## Example of Bad Output

```
🧠 **JUDGING**: Manager (user_task): Research sexual health and wellness interventions for: sildenafil mechanism  ##...

🧠 **JUDGING**: Manager (task_ledger):  We are working to address the following user request:  Research sexual healt...

🧠 **JUDGING**: Manager (instruction): Conduct targeted searches on PubMed, ClinicalTrials.gov, and Europe PMC to ga...
```

Users see:
- Raw internal prompts being passed between manager and agents
- Truncated text that cuts off mid-word ("healt...", "ga...")
- Technical jargon ("task_ledger") with no context
- All events labeled as "JUDGING" even when they're task assignments

## Root Cause Analysis

### The Chain of Issues

| Location | Issue |
|----------|-------|
| `src/orchestrators/advanced.py:363-370` | `MagenticOrchestratorMessageEvent` raw events exposed without filtering |
| `src/orchestrators/advanced.py:368` | `event.kind` values (`user_task`, `task_ledger`, `instruction`) are internal framework concepts |
| `src/orchestrators/advanced.py:368` | Hard truncation: `text[:200]...` breaks mid-sentence |
| `src/orchestrators/advanced.py:367` | All manager events mapped to `type="judging"` regardless of actual purpose |
| `src/orchestrators/advanced.py:380` | Agent messages also truncated at 200 chars |
| `src/utils/models.py:136` | `"judging": "🧠"` icon shown for all these internal events |
| `src/app.py:248` | Events displayed verbatim via `event.to_markdown()` |

### Code Path

```
agent-framework-core (Microsoft)
        ↓
MagenticOrchestratorMessageEvent(kind="task_ledger", message="...")
        ↓
advanced.py:_process_event() - NO FILTERING
        ↓
AgentEvent(type="judging", message=f"Manager ({event.kind}): {text[:200]}...")
        ↓
models.py:to_markdown() → "🧠 **JUDGING**: Manager (task_ledger): ..."
        ↓
app.py → Displayed to user verbatim
```

## Impact

1. **User Confusion**: Users see internal framework bookkeeping, not meaningful progress
2. **Truncated Gibberish**: 200-char limit cuts prompts mid-sentence, making them uninterpretable
3. **Misleading Labels**: "JUDGING" event type is wrong - these are task assignments
4. **No Actionable Info**: Users can't understand what the system is actually doing

## Proposed Solutions

### Option A: Filter Internal Events (Minimal)

Skip internal manager events entirely - they're framework bookkeeping:

```python
def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
    if isinstance(event, MagenticOrchestratorMessageEvent):
        # Skip internal framework bookkeeping events
        if event.kind in ("user_task", "task_ledger", "instruction"):
            return None  # Don't expose to users
        # ... rest of handling
```

**Pros**: Simple, removes noise
**Cons**: Users lose visibility into manager activity

### Option B: Transform to User-Friendly Messages (Better UX)

Map internal events to meaningful user messages:

```python
MANAGER_EVENT_MESSAGES = {
    "user_task": "Manager received research task",
    "task_ledger": "Manager tracking task progress",
    "instruction": "Manager assigning work to agent",
}

def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
    if isinstance(event, MagenticOrchestratorMessageEvent):
        if event.kind in MANAGER_EVENT_MESSAGES:
            return AgentEvent(
                type="progress",  # Not "judging"!
                message=MANAGER_EVENT_MESSAGES[event.kind],
                iteration=iteration,
            )
```

**Pros**: Users see meaningful progress, correct event types
**Cons**: More code, loses raw detail for debugging

### Option C: Smart Truncation + Verbose Mode

1. Truncate at sentence boundaries, not hard character limit
2. Add `verbose_mode` setting that shows full internal events for debugging
3. Use appropriate event types based on `event.kind`

```python
def _smart_truncate(self, text: str, max_len: int = 200) -> str:
    """Truncate at sentence boundary."""
    if len(text) <= max_len:
        return text
    # Find last sentence boundary before limit
    truncated = text[:max_len]
    last_period = truncated.rfind(". ")
    if last_period > max_len // 2:
        return truncated[:last_period + 1]
    return truncated.rsplit(" ", 1)[0] + "..."
```

### Recommended Approach

**Combine Option A + B**:

1. **Default**: Filter out `task_ledger` and `instruction` events (pure bookkeeping)
2. **Transform**: `user_task` → "Assigning research task to agents"
3. **Proper Types**: Use `"progress"` not `"judging"` for manager events
4. **Future**: Add verbose mode for debugging

## Files to Modify

1. `src/orchestrators/advanced.py:361-410` - `_process_event()` method
2. `src/utils/models.py:107-123` - Add new event types if needed
3. `tests/unit/orchestrators/test_advanced_timeout.py` - Update assertions

## Related Issues

- P0: Advanced Mode Timeout No Synthesis (FIXED in PR #104)
- This P1 was discovered while testing the P0 fix

## Testing the Bug

```python
import asyncio
from src.orchestrators.advanced import AdvancedOrchestrator

async def test():
    orch = AdvancedOrchestrator(max_rounds=3)
    async for event in orch.run("sildenafil mechanism"):
        if "Manager" in event.message:
            print(f"[{event.type}] {event.message}")
            # You'll see uninterpretable output

asyncio.run(test())
```

## References

- Microsoft Agent Framework: https://github.com/microsoft/agent-framework
- AgentEvent model: `src/utils/models.py:104`
- Advanced orchestrator: `src/orchestrators/advanced.py`
