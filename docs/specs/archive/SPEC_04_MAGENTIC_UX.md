# SPEC 04: Magentic Mode UX Improvements

## Priority: P1 (Demo Quality)

## Problem Statement

Magentic (advanced) mode has several UX issues that degrade the user experience:

1. **P0: Chat history cleared on timeout** - When timeout occurs, all progress events are erased
2. **P1: Timeout too short** - 300s default insufficient for complex multi-agent workflows
3. **P1: Timeout not configurable** - Users can't adjust based on their needs
4. **P2: No graceful degradation** - System doesn't synthesize early when timeout approaches

## Related Issues

- GitHub Issue #68: Magentic mode times out at 300s without completing
- GitHub Issue #65: Demo timing (predecessor, now closed)
- SPEC_01: Demo Termination (implemented the basic timeout)

## Bug Analysis

### Bug 1: Chat History Cleared on Timeout (P0)

**Location**: `src/app.py:205-206`

**Current Code**:
```python
if event.type == "complete":
    yield event.message  # BUG: Discards all accumulated progress!
else:
    event_md = event.to_markdown()
    response_parts.append(event_md)
    yield "\n\n".join(response_parts)
```

**Problem**: The `complete` event (including timeout) yields ONLY the completion message, discarding all the `response_parts` that show what the system actually did.

**User Sees**:
```
Research timed out. Synthesizing available evidence...
```

**User Should See**:
```
🚀 STARTED: Starting research (Magentic mode)...
⏳ THINKING: Multi-agent reasoning in progress...
🧠 JUDGING: Manager (user_task): Research drug repurposing...
🧠 JUDGING: Manager (task_ledger): We are working to address...
🧠 JUDGING: Manager (instruction): Task: Retrieve human clinical...
⏱️ Research timed out. Synthesizing available evidence...
```

**Fix**:
```python
if event.type == "complete":
    response_parts.append(event.message)
    yield "\n\n".join(response_parts)  # Preserves all progress
```

### Bug 2: Timeout Too Short (P1)

**Location**: `src/orchestrator_magentic.py:48`

**Current**: `timeout_seconds: float = 300.0` (5 minutes)

**Problem**: Multi-agent workflows with 4 agents (Search, Hypothesis, Judge, Report) and up to 10 rounds can theoretically take 60+ minutes. Even typical runs take 5-10 minutes.

**Analysis of Per-Agent Latency**:
| Agent | Typical Latency | Worst Case |
|-------|-----------------|------------|
| SearchAgent | 30-60s | 120s (network issues) |
| HypothesisAgent | 60-90s | 180s (complex reasoning) |
| JudgeAgent | 30-60s | 120s |
| ReportAgent | 60-120s | 240s (long synthesis) |

With `max_rounds=10`: 10 × 4 × 90s = 60 minutes worst case.

### Bug 3: Timeout Not Configurable (P1)

**Problem**: The factory doesn't pass timeout config to MagenticOrchestrator.

**Location**: `src/orchestrator_factory.py:52-55`
```python
return orchestrator_cls(
    max_rounds=config.max_iterations if config else 10,
    api_key=api_key,
    # Missing: timeout_seconds
)
```

## Proposed Solutions

### Fix 1: Preserve Chat History (P0)

```python
# src/app.py - Replace lines 205-212
if event.type == "complete":
    # Preserve accumulated progress + add completion message
    response_parts.append(event.message)
    yield "\n\n".join(response_parts)
else:
    event_md = event.to_markdown()
    response_parts.append(event_md)
    yield "\n\n".join(response_parts)
```

**Test**:
```python
@pytest.mark.asyncio
async def test_timeout_preserves_chat_history(mock_magentic_workflow):
    """Verify timeout doesn't erase progress events."""
    # Mock workflow that yields events then times out
    events = []
    async for event in research_agent("test", [], "advanced", "sk-test"):
        events.append(event)

    # Should contain both progress AND timeout message
    output = events[-1]  # Final yield
    assert "STARTED" in output
    assert "timed out" in output.lower()
```

### Fix 2: Increase Default Timeout (P1)

```python
# src/orchestrator_magentic.py
def __init__(
    self,
    max_rounds: int = 10,
    chat_client: OpenAIChatClient | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 600.0,  # Changed: 10 minutes (was 5)
) -> None:
```

### Fix 3: Make Timeout Configurable via Environment (P1)

```python
# src/utils/config.py
class Settings(BaseSettings):
    # ... existing fields ...
    magentic_timeout: int = Field(
        default=600,
        description="Timeout for Magentic mode in seconds",
    )
```

```python
# src/orchestrator_factory.py
return orchestrator_cls(
    max_rounds=config.max_iterations if config else 10,
    api_key=api_key,
    timeout_seconds=settings.magentic_timeout,  # NEW
)
```

### Fix 4: Graceful Degradation (P2 - Future)

```python
# src/orchestrator_magentic.py - Inside run() loop
elapsed = time.time() - start_time
time_remaining = self._timeout_seconds - elapsed

# If 80% of time elapsed, force synthesis
if time_remaining < self._timeout_seconds * 0.2:
    yield AgentEvent(
        type="synthesizing",
        message="Time limit approaching, synthesizing available evidence...",
        iteration=iteration,
    )
    # TODO: Inject signal to trigger ReportAgent
    break
```

## Implementation Order

1. **Fix 1 (P0)**: Chat history preservation - 5 minutes, 1 line change
2. **Fix 2 (P1)**: Increase default timeout - 5 minutes, 1 line change
3. **Fix 3 (P1)**: Environment config - 15 minutes, 3 files
4. **Fix 4 (P2)**: Graceful degradation - 1 hour, research agent-framework signals

## Acceptance Criteria

- [x] Timeout shows ALL progress events, not just timeout message
- [x] Default timeout increased to 600s (10 minutes)
- [x] Timeout configurable via `MAGENTIC_TIMEOUT` env var
- [x] Tests verify chat history preserved on timeout
- [ ] (P2) System synthesizes early when timeout approaches (Future)

**Status: IMPLEMENTED** (commit cb46aac)

## Files to Modify

1. `src/app.py` - Fix chat history clearing (lines 205-212)
2. `src/orchestrator_magentic.py` - Increase default timeout
3. `src/utils/config.py` - Add `magentic_timeout` setting
4. `src/orchestrator_factory.py` - Pass timeout to MagenticOrchestrator
5. `tests/unit/test_app_timeout.py` - NEW: Test chat history preservation

## Test Plan

```python
# tests/unit/test_app_timeout.py

@pytest.mark.asyncio
async def test_complete_event_preserves_history():
    """Complete events should append to history, not replace it."""
    from src.app import research_agent

    # This requires mocking the orchestrator to emit events then complete
    # Verify final output contains ALL events, not just completion message
    pass


@pytest.mark.asyncio
async def test_timeout_configurable():
    """Verify MAGENTIC_TIMEOUT env var is respected."""
    import os
    os.environ["MAGENTIC_TIMEOUT"] = "120"

    from src.utils.config import Settings
    settings = Settings()
    assert settings.magentic_timeout == 120
```

## Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| Fix 1 | Low | Simple change, well-understood |
| Fix 2 | Low | Just a default value change |
| Fix 3 | Medium | New config, needs validation |
| Fix 4 | High | Requires understanding agent-framework internals |

## Dependencies

- Fix 4 requires investigation of `agent-framework-core` to understand how to signal early termination to the workflow manager.
