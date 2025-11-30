# SPEC_15: Advanced Mode Performance Optimization

**Status**: Draft (Validated - Implement All Solutions)
**Priority**: P1
**GitHub Issue**: #65
**Estimated Effort**: Medium (config changes + early termination logic)
**Last Updated**: 2025-11-30

> **Senior Review Verdict**: ✅ APPROVED
> **Recommendation**: Implement Solution A + B + C together. Solution B (Early Termination) is NOT "post-hackathon" - it's the core fix that solves the root cause. The patterns used are consistent with Microsoft Agent Framework best practices.

---

## Problem Statement

Advanced (Multi-Agent) mode runs **10 rounds of multi-agent coordination** which takes **10-15+ minutes**.

**For hackathon demos**: No judge will wait this long. They'll close the tab before seeing results.

### Observed Behavior

- System works correctly (no crashes)
- Produces detailed, high-quality research output
- Takes too long for practical demo use
- User had to manually terminate after ~10 minutes

### Current Configuration

```python
# src/orchestrators/advanced.py:133
.with_standard_manager(
    chat_client=manager_client,
    max_round_count=self._max_rounds,  # Default: 10
    max_stall_count=3,
    max_reset_count=2,
)
```

### Time Breakdown (Estimated)

| Component | Time per Round | Notes |
|-----------|---------------|-------|
| Manager LLM call | 2-5s | Decides next agent |
| Search Agent | 10-20s | 4 API calls (PubMed, CT, EPMC, OA) |
| Hypothesis Agent | 5-10s | LLM reasoning |
| Judge Agent | 5-10s | LLM evaluation |
| Report Agent | 10-20s | LLM synthesis (when called) |

**Total per round**: ~30-60 seconds
**10 rounds**: 5-10 minutes minimum

---

## Root Cause Analysis

### Issue 1: Default `max_rounds=10` is Too High

The Microsoft Agent Framework keeps iterating until:
1. `max_rounds` reached, OR
2. Manager decides workflow is complete

For research tasks, the manager often wants "more evidence" and keeps searching.

### Issue 2: No Early Termination Heuristic

Even when the Judge says `sufficient=True` with high confidence, the workflow continues because the manager wants to be thorough.

### Issue 3: No User Expectation Setting

Users don't know how long to expect. Progress indication is minimal.

---

## Proposed Solutions

### Solution A: Reduce Default `max_rounds` (QUICK FIX)

**Change**: Reduce `max_rounds` from 10 to 5 (or make configurable via env).

```python
# src/orchestrators/advanced.py

def __init__(
    self,
    max_rounds: int | None = None,  # Changed from 10
    ...
) -> None:
    # Read from environment, default to 5 for faster demos
    default_rounds = int(os.getenv("ADVANCED_MAX_ROUNDS", "5"))
    self._max_rounds = max_rounds if max_rounds is not None else default_rounds
```

**Pros**:
- Simple, 2-line change
- Immediately halves demo time

**Cons**:
- Less thorough research
- Trade-off: speed vs. quality

### Solution B: Early Termination on High-Confidence Judge (RECOMMENDED)

**Change**: Add workflow termination signal when Judge returns `sufficient=True` with confidence > 70%.

This requires modifying the JudgeAgent to signal completion:

```python
# src/agents/magentic_agents.py - create_judge_agent()

@chat_agent.on_message
async def handle_judge_message(message: str, context: Context) -> ChatMessage:
    """Process judge request and potentially signal completion."""
    # ... existing judge logic ...

    assessment = await judge_handler.evaluate(evidence, query)

    if assessment.sufficient and assessment.confidence >= 0.70:
        # Signal to manager that we have enough evidence
        # The manager prompt should respect this signal
        return ChatMessage(
            content=f"SUFFICIENT EVIDENCE (confidence: {assessment.confidence:.0%}). "
            f"Recommend immediate synthesis. {assessment.reasoning}",
            metadata={"sufficient": True, "confidence": assessment.confidence},
        )

    return ChatMessage(content=f"INSUFFICIENT: {assessment.reasoning}")
```

And update the manager's system prompt to respect this:

```python
# src/orchestrators/advanced.py - _build_workflow()

manager_system_prompt = """You are a research workflow manager.

IMPORTANT: When JudgeAgent returns "SUFFICIENT EVIDENCE", immediately
delegate to ReportAgent for final synthesis. Do NOT continue searching.

Workflow:
1. SearchAgent finds evidence
2. HypothesisAgent generates hypotheses
3. JudgeAgent evaluates sufficiency
4. IF sufficient → ReportAgent synthesizes (END)
5. IF insufficient → SearchAgent refines search (CONTINUE)
"""
```

**Pros**:
- Respects actual evidence quality
- Can terminate early (round 3-4) when evidence is strong
- Maintains quality for complex queries

**Cons**:
- Requires testing to ensure manager respects signal
- More complex change

### Solution C: Better Progress Indication

Add estimated time remaining to UI:

```python
# src/orchestrators/advanced.py - run()

yield AgentEvent(
    type="progress",
    message=f"Round {iteration}/{self._max_rounds} "
            f"(~{(self._max_rounds - iteration) * 45}s remaining)",
    iteration=iteration,
)
```

**Pros**:
- Sets user expectations
- Doesn't change workflow behavior

**Cons**:
- Doesn't actually speed up the workflow

---

## Recommended Implementation

**IMPLEMENT ALL THREE SOLUTIONS NOW**:

1. **Solution A**: Reduce `max_rounds` to 5 via environment variable
2. **Solution B**: Early termination when Judge returns `sufficient=True` with confidence ≥70%
3. **Solution C**: Better progress indication with time estimates

> **Why Solution B NOW?** The Manager acting as a "termination condition" based on Judge feedback is a standard multi-agent pattern (Critique/Refine loop with exit). This aligns with Microsoft Agent Framework best practices and solves the ROOT CAUSE, not just a symptom.

---

## Implementation Details

### Phase 1: All Solutions Together (A + B + C)

#### 1. Update Advanced Orchestrator Constructor

```python
# src/orchestrators/advanced.py

import os

class AdvancedOrchestrator(OrchestratorProtocol):
    def __init__(
        self,
        max_rounds: int | None = None,
        chat_client: OpenAIChatClient | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,  # Reduced from 600 to 5 min
        domain: ResearchDomain | str | None = None,
    ) -> None:
        # Environment-configurable rounds (default 5 for demos)
        default_rounds = int(os.getenv("ADVANCED_MAX_ROUNDS", "5"))
        self._max_rounds = max_rounds if max_rounds is not None else default_rounds
        self._timeout_seconds = timeout_seconds
        # ... rest unchanged ...
```

#### 2. Add Progress Estimation

```python
# src/orchestrators/advanced.py - run()

# After processing MagenticAgentMessageEvent:
if isinstance(event, MagenticAgentMessageEvent):
    iteration += 1
    rounds_remaining = self._max_rounds - iteration
    # Estimate ~45s per round based on observed timing
    est_seconds = rounds_remaining * 45
    est_display = f"{est_seconds // 60}m {est_seconds % 60}s" if est_seconds >= 60 else f"{est_seconds}s"

    yield AgentEvent(
        type="progress",
        message=f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)",
        iteration=iteration,
    )
```

#### 3. Update UI Message (Solution C)

```python
# src/orchestrators/advanced.py - run()

# UX FIX: More accurate timing message
yield AgentEvent(
    type="thinking",
    message=(
        f"Multi-agent reasoning in progress ({self._max_rounds} rounds max)... "
        f"Estimated time: {self._max_rounds * 45 // 60}-{self._max_rounds * 60 // 60} minutes."
    ),
    iteration=0,
)
```

#### 4. Add Early Termination Signal (Solution B)

```python
# src/agents/magentic_agents.py - Update create_judge_agent()

@chat_agent.on_message
async def handle_judge_message(message: str, context: Context) -> ChatMessage:
    """Process judge request and signal completion when evidence is sufficient."""
    # ... existing parsing logic to extract evidence and query ...

    assessment = await judge_handler.evaluate(evidence, query)

    # NEW: Strong termination signal for high-confidence assessments
    if assessment.sufficient and assessment.confidence >= 0.70:
        # Clear, unambiguous signal that Manager should respect
        return ChatMessage(
            content=(
                f"✅ SUFFICIENT EVIDENCE (confidence: {assessment.confidence:.0%}). "
                f"STOP SEARCHING. Delegate to ReportAgent NOW for final synthesis. "
                f"Reasoning: {assessment.reasoning}"
            ),
            metadata={"sufficient": True, "confidence": assessment.confidence},
        )

    # Insufficient - continue the loop
    return ChatMessage(
        content=(
            f"❌ INSUFFICIENT: {assessment.reasoning}. "
            f"Confidence: {assessment.confidence:.0%}. "
            f"Suggested refinements: {', '.join(assessment.next_search_queries[:2])}"
        )
    )
```

#### 5. Update Manager System Prompt (Solution B)

```python
# src/orchestrators/advanced.py - _build_workflow()

MANAGER_SYSTEM_PROMPT = """You are a medical research workflow manager.

## CRITICAL RULE
When JudgeAgent says "SUFFICIENT EVIDENCE" or "STOP SEARCHING":
→ IMMEDIATELY delegate to ReportAgent for synthesis
→ Do NOT continue searching or gathering more evidence
→ The Judge has determined evidence quality is adequate

## Standard Workflow
1. SearchAgent → finds evidence from PubMed, ClinicalTrials, etc.
2. HypothesisAgent → generates testable hypotheses
3. JudgeAgent → evaluates evidence sufficiency
4. IF sufficient → ReportAgent (DONE)
5. IF insufficient → SearchAgent with refined queries (CONTINUE)

## Your Role
- Coordinate agents efficiently
- Respect the Judge's termination signal
- Prioritize completing the task over perfection
"""
```

---

## Test Plan

### Unit Tests

```python
# tests/unit/orchestrators/test_advanced_orchestrator.py

import os
from unittest.mock import patch

import pytest

from src.orchestrators.advanced import AdvancedOrchestrator


class TestAdvancedOrchestratorConfig:
    """Tests for configuration options."""

    def test_default_max_rounds_is_five(self) -> None:
        """Default max_rounds should be 5 for faster demos."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env var
            os.environ.pop("ADVANCED_MAX_ROUNDS", None)
            orch = AdvancedOrchestrator.__new__(AdvancedOrchestrator)
            orch.__init__()
            assert orch._max_rounds == 5

    def test_max_rounds_from_env(self) -> None:
        """max_rounds should be configurable via environment."""
        with patch.dict(os.environ, {"ADVANCED_MAX_ROUNDS": "3"}):
            orch = AdvancedOrchestrator.__new__(AdvancedOrchestrator)
            orch.__init__()
            assert orch._max_rounds == 3

    def test_explicit_max_rounds_overrides_env(self) -> None:
        """Explicit parameter should override environment."""
        with patch.dict(os.environ, {"ADVANCED_MAX_ROUNDS": "3"}):
            orch = AdvancedOrchestrator.__new__(AdvancedOrchestrator)
            orch.__init__(max_rounds=7)
            assert orch._max_rounds == 7

    def test_timeout_default_is_five_minutes(self) -> None:
        """Default timeout should be 300s (5 min) for faster failure."""
        orch = AdvancedOrchestrator.__new__(AdvancedOrchestrator)
        orch.__init__()
        assert orch._timeout_seconds == 300.0
```

### Integration Test (Manual)

```bash
# Run advanced mode with reduced rounds
ADVANCED_MAX_ROUNDS=3 uv run python -c "
import asyncio
from src.orchestrators.advanced import AdvancedOrchestrator

async def test():
    orch = AdvancedOrchestrator()
    print(f'Max rounds: {orch._max_rounds}')  # Should be 3

    async for event in orch.run('sildenafil mechanism'):
        print(f'{event.type}: {event.message[:100]}...')

asyncio.run(test())
"
```

### Timing Benchmark

Create a benchmark script to measure actual performance:

```python
# examples/benchmark_advanced.py
"""Benchmark Advanced mode with different max_rounds settings."""

import asyncio
import os
import time


async def benchmark(max_rounds: int) -> float:
    """Run benchmark with specified rounds, return elapsed time."""
    os.environ["ADVANCED_MAX_ROUNDS"] = str(max_rounds)

    # Import after setting env
    from src.orchestrators.advanced import AdvancedOrchestrator

    orch = AdvancedOrchestrator()
    start = time.time()

    async for event in orch.run("sildenafil erectile dysfunction"):
        if event.type == "complete":
            break

    return time.time() - start


async def main() -> None:
    """Run benchmarks for different configurations."""
    for rounds in [3, 5, 7, 10]:
        elapsed = await benchmark(rounds)
        print(f"max_rounds={rounds}: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/orchestrators/advanced.py` | Add env-configurable `max_rounds`, reduce default to 5, add progress estimation, update Manager prompt |
| `src/agents/magentic_agents.py` | Add early termination signal in JudgeAgent |
| `tests/unit/orchestrators/test_advanced_orchestrator.py` | Add config tests |
| `tests/unit/agents/test_magentic_judge_termination.py` | Add termination signal tests |
| `examples/benchmark_advanced.py` | Add timing benchmark (optional) |

---

## Acceptance Criteria

### Solution A: Configuration
- [ ] Default `max_rounds` is 5 (not 10)
- [ ] `max_rounds` configurable via `ADVANCED_MAX_ROUNDS` env var
- [ ] Explicit `max_rounds` parameter overrides env var
- [ ] Default timeout is 5 minutes (300s, not 600s)

### Solution B: Early Termination
- [ ] JudgeAgent returns "SUFFICIENT EVIDENCE" message when confidence ≥70%
- [ ] JudgeAgent returns "STOP SEARCHING" in termination signal
- [ ] Manager system prompt includes explicit termination instructions
- [ ] Workflow terminates early when Judge signals sufficiency (observed in logs)

### Solution C: Progress Indication
- [ ] Progress events show current round / max rounds
- [ ] Progress events show estimated time remaining
- [ ] Initial "thinking" message shows estimated total time

### Overall
- [ ] Demo completes in <5 minutes with useful output
- [ ] Quality of output is maintained (no degradation from early termination)

---

## Rollback Plan

If reduced rounds cause quality issues:
1. Increase `ADVANCED_MAX_ROUNDS` environment variable
2. No code changes needed

---

## References

- GitHub Issue #65
- Microsoft Agent Framework: https://github.com/microsoft/agent-framework
- MagenticBuilder docs: Round configuration
