# Architectural Debt Specification

> **Status**: IMPLEMENTED (Phase 1 Complete)
> **Date**: 2025-12-04
> **Scope**: `src/orchestrators/advanced.py` (Primary), System-Wide (Secondary)
> **Purpose**: Roadmap for "DeepMind Status" Code Quality
> **Author**: Claude (Senior Review Incorporated)

---

## Executive Summary

The P1/P2 bug fixes in PR #124 introduced technical debt that must be addressed before the PR is considered "done". This spec documents three immediate priorities (DRY violation, redundant imports, magic strings) and five medium-term system-wide improvements.

---

## Part 1: Immediate Cleanup (MUST Complete Before PR Merge)

### Priority 1: DRY Violation - Synthesis Methods

**Problem**: `_handle_timeout()` (lines 201-248) and `_force_synthesis()` (lines 250-297) are **95% identical**.

| `_handle_timeout()` | `_force_synthesis()` |
|---------------------|----------------------|
| Lines 201-248 (47 lines) | Lines 250-297 (47 lines) |
| Yields "Workflow timed out. Synthesizing..." | Yields "Synthesizing research findings..." |
| Error data: `timeout_synthesis` | Error data: `forced_synthesis` |
| **Everything else is identical** | **Everything else is identical** |

**SOLID Violation**: **DRY (Don't Repeat Yourself)**. Changes to synthesis logic must be made in two places. This is a maintenance nightmare and a source of future bugs.

**Fix**: Extract unified method `_synthesize_fallback(iteration: int, reason: str)`:

```python
async def _synthesize_fallback(
    self, iteration: int, reason: str
) -> AsyncGenerator[AgentEvent, None]:
    """
    Unified fallback synthesis for all termination scenarios.

    Args:
        iteration: Current workflow iteration
        reason: Why synthesis is being forced ("timeout", "no_reporter", "max_rounds")
    """
    status_messages = {
        "timeout": "Workflow timed out. Synthesizing available evidence...",
        "no_reporter": "Synthesizing research findings...",
        "max_rounds": "Max rounds reached. Synthesizing findings...",
    }

    try:
        state = get_magentic_state()
        evidence_summary = await state.memory.get_context_summary()
        report_agent = create_report_agent(self._chat_client, domain=self.domain)

        yield AgentEvent(
            type="synthesizing",
            message=status_messages.get(reason, "Synthesizing..."),
            iteration=iteration,
        )

        synthesis_result = await report_agent.run(
            f"Synthesize research report from this evidence. "
            f"If evidence is sparse, say so.\n\n{evidence_summary}"
        )

        yield AgentEvent(
            type="complete",
            message=synthesis_result.text,
            data={"reason": f"{reason}_synthesis", "iterations": iteration},
            iteration=iteration,
        )
    except Exception as synth_error:
        logger.error(f"{reason} synthesis failed", error=str(synth_error))
        yield AgentEvent(
            type="complete",
            message=f"Research completed. Synthesis failed: {synth_error}",
            data={"reason": f"{reason}_synthesis_failed", "iterations": iteration},
            iteration=iteration,
        )
```

**Call Sites to Update**:
1. Line 447: `async for event in self._handle_timeout(iteration):` → `self._synthesize_fallback(iteration, "timeout")`
2. Line 412: `async for synth_event in self._force_synthesis(iteration):` → `self._synthesize_fallback(iteration, "no_reporter")`
3. Line 432: `async for synth_event in self._force_synthesis(iteration):` → `self._synthesize_fallback(iteration, "max_rounds")`

**Delete After Refactor**:
- `_handle_timeout()` method (lines 201-248)
- `_force_synthesis()` method (lines 250-297)

---

### Priority 2: Redundant Imports

**Problem**: Imports inside methods that already exist at module level.

| Location | Import | Already At |
|----------|--------|------------|
| Line 207 | `from src.agents.magentic_agents import create_report_agent` | Line 35 |
| Line 208 | `from src.agents.state import get_magentic_state` | Missing! |
| Line 257 | `from src.agents.magentic_agents import create_report_agent` | Line 35 |
| Line 258 | `from src.agents.state import get_magentic_state` | Missing! |

**SOLID Violation**: **SRP (Single Responsibility)**. Import management is scattered across the file instead of centralized at the top.

**Fix**:
1. Add to module-level imports (around line 38):
   ```python
   from src.agents.state import get_magentic_state, init_magentic_state
   ```
   Note: `init_magentic_state` is already imported at line 38. Add `get_magentic_state` to that import.

2. Remove redundant imports from:
   - Lines 207-208 (inside `_handle_timeout`)
   - Lines 257-258 (inside `_force_synthesis`)

---

### Priority 3: Magic Strings

**Problem**: Agent detection relies on string literals that break silently if agents are renamed.

**Current Code** (line 385):
```python
agent_name = (event.agent_id or "").lower()
if "report" in agent_name:  # FRAGILE: Breaks if agent renamed
    reporter_ran = True
```

**Also in** `_get_event_type_for_agent()` (lines 593-602):
```python
if "search" in agent_lower:    # Magic string
if "judge" in agent_lower:     # Magic string
if "hypothes" in agent_lower:  # Magic string
if "report" in agent_lower:    # Magic string
```

**SOLID Violation**: **OCP (Open/Closed Principle)**. Renaming an agent requires changes in multiple locations.

**Fix Option A** - Constants:
```python
# At module level (after imports)
REPORTER_AGENT_ID = "reporter"
SEARCHER_AGENT_ID = "searcher"
JUDGE_AGENT_ID = "judge"
HYPOTHESIZER_AGENT_ID = "hypothesizer"
```

**Fix Option B** - Agent Name Attribute (Preferred):
```python
# In magentic_agents.py, ensure each agent has a .name attribute
# Then in advanced.py:
if event.agent_id == report_agent.name:
    reporter_ran = True
```

**Recommendation**: Option A is simpler and doesn't require modifying agent factory. Use constants.

---

## Part 2: System-Wide Issues (Future PRs)

These are valid concerns identified during code review but are NOT blockers for the current PR.

### Priority 4: Dead Config

**Location**: `src/utils/config.py`
**Issue**: Zombie configuration values that are never used or raise NotImplemented.
- `magentic_timeout`: Deprecated, never read
- `anthropic_api_key`: Config exists but factory raises NotImplemented

**Fix**: Audit config.py, remove dead settings, add deprecation warnings for transitional settings.

---

### Priority 5: Prompt Unification

**Location**: `src/prompts/` vs `src/config/domain.py`
**Issue**: Two sources of truth for prompts. `src/prompts/` files exist but are ignored. System uses hardcoded strings in `domain.py`.

**Fix**: Pick ONE source of truth. Recommendation: Delete `src/prompts/` if unused, or migrate `domain.py` prompts there.

---

### Priority 6: Factory Monolith

**Location**: `src/clients/factory.py`
**Issue**: Hardcoded logic for detecting API key prefixes (`sk-` → OpenAI, `sk-ant-` → Anthropic error).

**SOLID Violation**: OCP. Adding a new provider requires modifying the factory.

**Fix**: Provider registry pattern with auto-registration, or strategy pattern with key prefix handlers.

---

### Priority 7: State Class

**Location**: `src/orchestrators/advanced.py` `run()` method
**Issue**: Method manages 6+ loose variables (`iteration`, `reporter_ran`, `buffer`, `current_agent_id`, `last_streamed_length`, `final_event_received`).

**Fix**: Extract to `WorkflowState` dataclass:
```python
@dataclass
class WorkflowState:
    iteration: int = 0
    reporter_ran: bool = False
    current_message_buffer: str = ""
    current_agent_id: str | None = None
    last_streamed_length: int = 0
    final_event_received: bool = False
```

---

### Priority 8: Real Integration Tests

**Location**: `tests/e2e/`
**Issue**: We deleted flaky integration tests. Now we have ZERO automated tests against real APIs.

**Fix**: Create stable `make test-live` suite with:
- Real HuggingFace Free Tier test
- Real OpenAI BYOK test
- Proper timeout handling
- Skip markers for CI (run manually or on schedule)

---

## Regression Prevention Strategy

**CRITICAL**: Each phase MUST pass smoke tests before merge. Unit tests alone are insufficient.

> **Implementation Status**: IMPLEMENTED (PR #125)
> Smoke tests are now live in `tests/e2e/test_smoke.py` with Makefile targets.
> Run `make smoke-free` or `make smoke-paid` before any refactoring PR.

### Smoke Test Infrastructure

Add to `Makefile`:
```makefile
# Smoke tests - run against real APIs (slow, not for CI)
smoke-free:
	@echo "Running Free Tier smoke test..."
	uv run python -m pytest tests/e2e/test_smoke.py::test_free_tier_synthesis -v -s --timeout=600

smoke-paid:
	@echo "Running Paid Tier smoke test (requires OPENAI_API_KEY)..."
	uv run python -m pytest tests/e2e/test_smoke.py::test_paid_tier_synthesis -v -s --timeout=300

smoke: smoke-free  # Default to free tier
```

### Smoke Test Implementation

Create `tests/e2e/test_smoke.py`:
```python
"""
Smoke tests for regression prevention.

These tests run against REAL APIs and verify end-to-end functionality.
They are slow (2-5 minutes) and should NOT run in CI.

Usage:
    make smoke-free   # Test Free Tier (HuggingFace)
    make smoke-paid   # Test Paid Tier (OpenAI BYOK)
"""
import pytest
from src.orchestrators.advanced import AdvancedOrchestrator

@pytest.mark.e2e
@pytest.mark.timeout(600)  # 10 minute timeout for Free Tier
async def test_free_tier_synthesis():
    """Verify Free Tier produces actual synthesis (not just 'Research complete.')"""
    orch = AdvancedOrchestrator(max_rounds=2)

    events = []
    async for event in orch.run("What is libido?"):
        events.append(event)

    # MUST have a complete event
    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) >= 1, "No complete event received"

    # Complete event MUST have substantive content (not just signal)
    final = complete_events[-1]
    assert len(final.message) > 100, f"Synthesis too short: {len(final.message)} chars"
    assert "Research complete." not in final.message or len(final.message) > 50, \
        "Got empty synthesis signal instead of actual report"

    # Should NOT have duplicate content
    messages = [e.message for e in events if e.message]
    # Check for exact duplicates of long content
    long_messages = [m for m in messages if len(m) > 200]
    assert len(long_messages) == len(set(long_messages)), "Duplicate content detected"

@pytest.mark.e2e
@pytest.mark.timeout(300)  # 5 minute timeout for Paid Tier
async def test_paid_tier_synthesis():
    """Verify Paid Tier (BYOK) produces synthesis."""
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    orch = AdvancedOrchestrator(max_rounds=2, api_key=api_key)

    events = []
    async for event in orch.run("What is libido?"):
        events.append(event)

    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) >= 1
    assert len(complete_events[-1].message) > 100
```

### Phase Gate Checklist

Before merging ANY refactoring PR:

```text
[ ] make check          # All 318+ unit tests pass
[ ] make smoke-free     # Free Tier produces real synthesis
[ ] make smoke-paid     # Paid Tier works (if you have key)
[ ] CodeRabbit approved # No blocking issues
```

---

## Execution Strategy

### Phase 1: Current PR (REQUIRED)
Implement **Priority 1, 2, and 3** before merging PR #124.

**Definition of Done**:
- [x] `_synthesize_fallback(iteration, reason)` implemented
- [x] `_handle_timeout()` and `_force_synthesis()` deleted
- [x] All synthesis call sites updated
- [x] Redundant imports removed
- [x] `get_magentic_state` added to module-level imports
- [x] Magic strings replaced with constants
- [x] All tests pass (`make check`)

### Phase 2: Future PRs (Separate Tickets)
Create GitHub issues for Priority 4-8. Do NOT bloat the current bug fix PR.

**IMPORTANT**: Before starting ANY Priority 4-7 refactors, FIRST implement Priority 8 (Smoke Tests).
This ensures we can detect regressions from refactoring. Sequence:

1. **PR: Smoke Test Infrastructure** (Priority 8) - MUST BE FIRST
   - Create `tests/e2e/test_smoke.py`
   - Add `make smoke-free` and `make smoke-paid` to Makefile
   - Verify both Free Tier and Paid Tier produce synthesis

2. **PR: Dead Config Cleanup** (Priority 4)
3. **PR: Prompt Unification** (Priority 5)
4. **PR: Factory Registry Pattern** (Priority 6)
5. **PR: WorkflowState Dataclass** (Priority 7)

---

## Appendix: Line Number Reference (Historical)

> **Note**: These line numbers were from BEFORE Phase 1 refactoring.
> After PR #124 merge, the following methods were consolidated:
> - `_handle_timeout()` → DELETED (merged into `_synthesize_fallback`)
> - `_force_synthesis()` → DELETED (merged into `_synthesize_fallback`)
> - Redundant imports → REMOVED (centralized at module level)
> - Magic strings → REPLACED with constants

| Item (Pre-Refactor) | Original Location | Status |
|---------------------|-------------------|--------|
| `_handle_timeout()` | Lines 201-248 | DELETED |
| `_force_synthesis()` | Lines 250-297 | DELETED |
| Redundant imports (timeout) | Lines 207-208 | REMOVED |
| Redundant imports (force) | Lines 257-258 | REMOVED |
| Magic string detection | Line 385 | REFACTORED |
| `_get_event_type_for_agent()` | Lines 582-602 | REFACTORED |
| Module imports | Lines 18-48 | UPDATED |
| `run()` method | Lines 299-456 | UPDATED |
