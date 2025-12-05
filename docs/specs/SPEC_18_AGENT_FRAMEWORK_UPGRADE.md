# SPEC-18: Agent Framework Core Upgrade Strategy

**Status**: APPROVED - Senior Review Complete
**Created**: 2025-12-04
**Priority**: P0 (Blocking HuggingFace Spaces deployment)
**Estimated Effort**: 2-4 hours (implementation + testing)

---

## Executive Summary

The `agent-framework-core` package released version `1.0.0b251204` on 2025-12-04 with breaking API changes. HuggingFace Spaces pulls this new version, breaking our app with:

```text
cannot import name 'MagenticAgentDeltaEvent' from 'agent_framework'
```

**Key Insight**: The upstream release **FIXED the repr bug** that we reported and worked around with our Accumulator Pattern (SPEC-17). This creates an opportunity to simplify our codebase.

**Recommendation**: UPGRADE to latest version (not pin to old) because:
1. The repr bug fix means we can simplify our Accumulator Pattern
2. Beta versions move fast - staying current is better long-term
3. The migration is smaller than initially estimated (~30 lines to change)

---

## Root Cause Analysis

### Why This Happened

1. **Dependency drift**: Our `requirements.txt` used `>=1.0.0b251120,<2.0.0` (range)
2. **New release**: Microsoft released `1.0.0b251204` on the SAME DAY (Dec 4, 2025)
3. **HuggingFace behavior**: Fresh installs pull latest matching version
4. **Breaking changes**: New version removed 3 event classes we depend on

### Pattern Recognition

This is the SECOND dependency-related P0 today:
1. **MCP ToolUseContent** - Same pattern (requirements.txt drift)
2. **Agent Framework events** - Same pattern (requirements.txt drift)

**Lesson**: Beta dependencies need EXACT pinning, not ranges.

---

## Version Comparison

| Version | Date | Status |
|---------|------|--------|
| `1.0.0b251120` | Nov 20, 2025 | Our current version (has repr bug) |
| `1.0.0b251204` | Dec 4, 2025 | Latest (fixes repr bug, breaking API) |

### What Microsoft Changed

From [GitHub releases](https://github.com/microsoft/agent-framework/releases):

1. **"Standardized orchestration outputs"** - Unified event system
2. **"Fixed MagenticAgentExecutor from producing repr string"** - Our bug is fixed!
3. **Event class refactoring** - Magentic-specific → Generic events

---

## API Changes (Complete Analysis)

### Classes REMOVED (Breaking)

```python
# OLD (1.0.0b251120) - These no longer exist
MagenticAgentDeltaEvent      # Streaming text chunks
MagenticAgentMessageEvent    # Agent turn completion
MagenticFinalResultEvent     # Final result
```

### Classes ADDED (Replacements)

```python
# NEW (1.0.0b251204) - Use these instead
AgentRunUpdateEvent          # Streaming updates
  └── .data: AgentRunResponseUpdate
           └── .text: str          # Streaming text (same data!)
           └── .contents: list     # Content blocks
           └── .author_name: str   # Agent ID (was .agent_id)
           └── .role: str          # Role (assistant, etc.)

ExecutorCompletedEvent       # Agent turn complete (CRITICAL!)
  └── .executor_id: str      # Which agent completed
  └── .data: Any             # Completion data

WorkflowOutputEvent          # Workflow finished (unchanged)
```

### Classes UNCHANGED

```python
# These work the same in both versions
MagenticBuilder              ✅
MagenticContext              ✅
MagenticOrchestratorMessageEvent  ✅  # Still exists!
WorkflowOutputEvent          ✅
BaseChatClient               ✅
ChatAgent                    ✅
ai_function                  ✅
```

### Migration Mapping

| OLD | NEW | Notes |
|-----|-----|-------|
| `MagenticAgentDeltaEvent.text` | `AgentRunUpdateEvent.data.text` | Same data, extra `.data` level |
| `MagenticAgentDeltaEvent.agent_id` | `AgentRunUpdateEvent.data.author_name` | Renamed field |
| `MagenticAgentMessageEvent` | `ExecutorCompletedEvent` | **CRITICAL**: Agent turn complete signal |
| `MagenticAgentMessageEvent.agent_id` | `ExecutorCompletedEvent.executor_id` | Agent identifier |
| `MagenticFinalResultEvent` | `WorkflowOutputEvent` | Workflow complete |
| `MagenticOrchestratorMessageEvent` | `MagenticOrchestratorMessageEvent` | UNCHANGED ✅ |

---

## Files Requiring Changes

### Critical (Must Change)

| File | Lines | Changes Required |
|------|-------|------------------|
| `src/orchestrators/advanced.py` | 24-31, 327-385 | Import updates + event handling |
| `tests/unit/orchestrators/test_accumulator_pattern.py` | 17-94 | Mock class updates |
| `tests/unit/orchestrators/test_advanced_events.py` | 4 | Import update |
| `requirements.txt` | 49 | Pin exact version |
| `pyproject.toml` | 64 | Pin exact version |

### Documentation (Update References)

| File | Status | Action |
|------|--------|--------|
| `docs/workflow-diagrams.md` | Has old class names | Update diagram labels |
| `docs/specs/archive/SPEC_17_ACCUMULATOR_PATTERN.md` | Archived | Add "Superseded" note |
| `docs/bugs/archive/P0_REPR_BUG_ROOT_CAUSE_ANALYSIS.md` | Archived | Add "Fixed upstream" note |
| `docs/architecture/system_registry.md` | Has outdated refs | Update tool inventory |

### Config Files (Pin Versions)

| File | Current | Change To |
|------|---------|-----------|
| `requirements.txt` | `>=1.0.0b251120,<2.0.0` | `==1.0.0b251204` |
| `pyproject.toml` | `>=1.0.0b251120,<2.0.0` | `==1.0.0b251204` |
| `uv.lock` | Auto-generated | Run `uv lock` after |

---

## Implementation Plan

### Phase 1: Version Pinning (5 min)
```bash
# Update requirements.txt and pyproject.toml
agent-framework-core==1.0.0b251204

# Regenerate lock file
uv lock
uv sync
```

### Phase 2: Import Updates (10 min)

**`src/orchestrators/advanced.py` lines 24-31:**
```python
# OLD
from agent_framework import (
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticOrchestratorMessageEvent,
    WorkflowOutputEvent,
)

# NEW
from agent_framework import (
    AgentRunUpdateEvent,        # Replaces MagenticAgentDeltaEvent
    ExecutorCompletedEvent,     # Replaces MagenticAgentMessageEvent (CRITICAL!)
    MagenticBuilder,
    MagenticOrchestratorMessageEvent,  # UNCHANGED
    WorkflowOutputEvent,        # Replaces MagenticFinalResultEvent
)
```

### Phase 3: Event Handling Refactor (30 min)

**`src/orchestrators/advanced.py` lines 326-385:**

```python
# ============================================
# 1. STREAMING: AgentRunUpdateEvent
# ============================================
# OLD:
if isinstance(event, MagenticAgentDeltaEvent):
    if event.text:
        state.current_message_buffer += event.text
        yield AgentEvent(type="streaming", message=event.text, ...)

# NEW:
if isinstance(event, AgentRunUpdateEvent) and event.data:
    author = getattr(event.data, 'author_name', None)
    if author != state.current_agent_id:
        state.current_message_buffer = ""
        state.current_agent_id = author

    text = event.data.text
    if text:
        state.current_message_buffer += text
        yield AgentEvent(type="streaming", message=text, data={"agent_id": author}, ...)
    continue

# ============================================
# 2. COMPLETION SIGNAL: ExecutorCompletedEvent (CRITICAL!)
# ============================================
# OLD:
if isinstance(event, MagenticAgentMessageEvent):
    state.iteration += 1
    # ... handle completion ...

# NEW:
if isinstance(event, ExecutorCompletedEvent):
    state.iteration += 1
    agent_name = event.executor_id or ""

    # Track if ReportAgent ran (for P1 forced synthesis)
    if REPORTER_AGENT_ID in agent_name.lower():
        state.reporter_ran = True

    # Use accumulated buffer (or trust event.data.text since repr bug is fixed)
    comp_event, prog_event = self._handle_completion_event(
        event, state.current_message_buffer, state.iteration
    )
    yield comp_event
    yield prog_event
    state.current_message_buffer = ""
    continue

# ============================================
# 3. FINAL EVENT: WorkflowOutputEvent (unchanged pattern)
# ============================================
if isinstance(event, WorkflowOutputEvent):
    # ... same as before, MagenticFinalResultEvent removed ...
```

**Key Simplification**: Since the repr bug is fixed upstream:
1. We MAY be able to trust `event.data.text` directly
2. Keep the buffer as safety net for this PR
3. Mark Accumulator Pattern for potential removal in next sprint

### Phase 4: Test Updates (30 min)

**`tests/unit/orchestrators/test_accumulator_pattern.py`:**
- Update mock classes to match new API
- Verify Accumulator Pattern still works OR
- Update tests if we simplify the pattern

**`tests/unit/orchestrators/test_advanced_events.py`:**
- Update import from `MagenticOrchestratorMessageEvent` (still exists)

### Phase 5: Verification (30 min)

```bash
# Run full test suite
make check

# Verify specific tests
uv run pytest tests/unit/orchestrators/ -v

# Manual smoke test
uv run python src/app.py
# Test free tier (HuggingFace)
# Test paid tier (OpenAI) if key available
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| New bugs in event handling | MEDIUM | HIGH | Comprehensive test coverage |
| Breaking streaming | LOW | HIGH | Keep buffer pattern as safety net |
| Other API changes we missed | LOW | MEDIUM | Check all imports against new version |
| HuggingFace caching old version | LOW | LOW | Use `--upgrade` flag |

---

## Rollback Plan

If upgrade fails:
1. Pin to OLD version: `agent-framework-core==1.0.0b251120`
2. Keep existing Accumulator Pattern
3. File issue with Microsoft if blocking bugs found

---

## Decision: Upgrade vs Pin Old

### Option A: Upgrade to `1.0.0b251204` ✅ RECOMMENDED

| Pros | Cons |
|------|------|
| Repr bug fixed upstream | Requires code changes |
| Simpler codebase (maybe remove workaround) | Testing effort |
| Aligned with upstream | Risk of new bugs |
| Future-proof | |

### Option B: Pin to `1.0.0b251120`

| Pros | Cons |
|------|------|
| Zero code changes | Technical debt (keep workaround forever) |
| Immediate fix | Missing bug fixes |
| Known behavior | Eventually unsupported |

**Recommendation**: **UPGRADE** - The effort is ~2-4 hours, but we get:
1. Upstream repr bug fix (no more workaround needed)
2. Cleaner codebase
3. Alignment with Microsoft's direction

---

## Success Criteria

- [ ] All 302 tests pass
- [ ] `make check` passes (lint, type, test)
- [ ] App starts on local
- [ ] Free tier works (HuggingFace backend)
- [ ] Paid tier works (OpenAI backend)
- [ ] Streaming works correctly
- [ ] No repr garbage in output
- [ ] HuggingFace Spaces deployment works

---

## Post-Upgrade Cleanup

After successful upgrade:
1. Update `docs/specs/archive/SPEC_17_ACCUMULATOR_PATTERN.md` - Mark as "Superseded by upstream fix"
2. Consider removing Accumulator Pattern if no longer needed
3. Update `docs/architecture/system_registry.md` with new event classes
4. Add regression test to prevent future dependency drift

---

## Appendix A: Full File Diff Preview

### `src/orchestrators/advanced.py` (estimated diff)

```diff
 from agent_framework import (
-    MagenticAgentDeltaEvent,
-    MagenticAgentMessageEvent,
+    AgentRunUpdateEvent,
+    ExecutorCompletedEvent,
     MagenticBuilder,
-    MagenticFinalResultEvent,
     MagenticOrchestratorMessageEvent,
     WorkflowOutputEvent,
 )

 # ... lines 300-325 unchanged ...

                     # 1. Handle Streaming
-                    if isinstance(event, MagenticAgentDeltaEvent):
-                        if event.agent_id != state.current_agent_id:
+                    if isinstance(event, AgentRunUpdateEvent) and event.data:
+                        author = getattr(event.data, 'author_name', None)
+                        if author != state.current_agent_id:
                             state.current_message_buffer = ""
-                            state.current_agent_id = event.agent_id
-
-                        if event.text:
-                            state.current_message_buffer += event.text
+                            state.current_agent_id = author
+
+                        text = event.data.text
+                        if text:
+                            state.current_message_buffer += text
                             yield AgentEvent(
                                 type="streaming",
-                                message=event.text,
-                                data={"agent_id": event.agent_id},
+                                message=text,
+                                data={"agent_id": author},
                                 iteration=state.iteration,
                             )
                         continue

                     # 2. Handle Completion Signal
-                    if isinstance(event, MagenticAgentMessageEvent):
+                    if isinstance(event, ExecutorCompletedEvent):
                         state.iteration += 1
-                        agent_name = (event.agent_id or "").lower()
+                        agent_name = (event.executor_id or "").lower()
                         if REPORTER_AGENT_ID in agent_name:
                             state.reporter_ran = True
                         # ... rest of completion handling ...
                         continue

                     # 3. Handle Final Events
-                    if isinstance(event, (MagenticFinalResultEvent, WorkflowOutputEvent)):
+                    if isinstance(event, WorkflowOutputEvent):
                         # ... final event handling (remove MagenticFinalResultEvent check) ...
```

---

## Appendix B: Lessons Learned

### Dependency Management Best Practices

1. **Beta packages need EXACT pins**: `==1.0.0b251204` not `>=1.0.0b251120`
2. **Sync requirements.txt and pyproject.toml**: HuggingFace uses requirements.txt
3. **Lock files are critical**: `uv.lock` captures exact versions
4. **Monitor upstream releases**: Set up GitHub watch on critical deps

### Why This Pattern Will Recur

Microsoft Agent Framework is in **active beta** development. Expect:
- Frequent releases (weekly/biweekly)
- Breaking changes without major version bumps
- API refactoring as they stabilize

**Strategy**: Pin exact versions, upgrade deliberately, not automatically.

---

## Senior Agent Review: COMPLETE

**Review conducted**: 2025-12-04

### Questions & Answers:

1. **Is the upgrade approach correct?**
   ✅ YES - Upgrade is recommended. Pinning to old version would accumulate tech debt.

2. **Are there any API changes we missed?**
   ✅ FOUND: `ExecutorCompletedEvent` was missing from original spec.
   - This is the replacement for `MagenticAgentMessageEvent` (agent turn completion)
   - **CRITICAL** - Must be handled for proper workflow control

3. **Should we keep the Accumulator Pattern as a safety net?**
   ✅ YES for this PR - Keep buffer as safety net, mark for potential removal in next sprint.
   - Upstream claims repr bug is fixed
   - Verify with smoke test before removing workaround

4. **Testing strategy for streaming behavior?**
   ✅ Must include manual smoke test in addition to unit tests.
   - Unit tests mock events, may not catch runtime issues
   - Run `python src/app.py` and verify tokens stream correctly

### Reviewer Notes:

> "The analysis is accurate. The move to 'Standardized orchestration outputs' by Microsoft
> explains the removal of Magentic-specific events in favor of generic AgentRun* events.
> Proceed with the upgrade, ensuring ExecutorCompletedEvent is properly handled."

**Status**: APPROVED FOR IMPLEMENTATION
