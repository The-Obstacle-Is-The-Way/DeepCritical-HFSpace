# Fix Plan: Magentic Mode Report Generation

**Related Bug**: `P0_MAGENTIC_MODE_BROKEN.md`
**Approach**: Test-Driven Development (TDD)
**Estimated Scope**: 4 tasks, ~2-3 hours

---

## Problem Summary

Magentic mode runs but fails to produce readable reports due to:

1. **Primary Bug**: `MagenticFinalResultEvent.message` returns `ChatMessage` object, not text
2. **Secondary Bug**: Max rounds (3) reached before ReportAgent completes
3. **Tertiary Issues**: Stale "bioRxiv" references in prompts

---

## Fix Order (TDD)

### Phase 1: Write Failing Tests

**Task 1.1**: Create test for ChatMessage text extraction

```python
# tests/unit/test_orchestrator_magentic.py

def test_process_event_extracts_text_from_chat_message():
    """Final result event should extract text from ChatMessage object."""
    # Arrange: Mock ChatMessage with .content attribute
    # Act: Call _process_event with MagenticFinalResultEvent
    # Assert: Returned AgentEvent.message is a string, not object repr
```

**Task 1.2**: Create test for max rounds configuration

```python
def test_orchestrator_uses_configured_max_rounds():
    """MagenticOrchestrator should use max_rounds from constructor."""
    # Arrange: Create orchestrator with max_rounds=10
    # Act: Build workflow
    # Assert: Workflow has max_round_count=10
```

**Task 1.3**: Create test for bioRxiv reference removal

```python
def test_task_prompt_references_europe_pmc():
    """Task prompt should reference Europe PMC, not bioRxiv."""
    # Arrange: Create orchestrator
    # Act: Check task string in run()
    # Assert: Contains "Europe PMC", not "bioRxiv"
```

---

### Phase 2: Fix ChatMessage Text Extraction

**File**: `src/orchestrator_magentic.py`
**Lines**: 192-199

**Current Code**:
```python
elif isinstance(event, MagenticFinalResultEvent):
    text = event.message.text if event.message else "No result"
```

**Fixed Code**:
```python
elif isinstance(event, MagenticFinalResultEvent):
    if event.message:
        # ChatMessage may have .content or .text depending on version
        if hasattr(event.message, 'content') and event.message.content:
            text = str(event.message.content)
        elif hasattr(event.message, 'text') and event.message.text:
            text = str(event.message.text)
        else:
            # Fallback: convert entire message to string
            text = str(event.message)
    else:
        text = "No result generated"
```

**Why**: The `agent_framework.ChatMessage` object structure may vary. We need defensive extraction.

---

### Phase 3: Fix Max Rounds Configuration

**File**: `src/orchestrator_magentic.py`
**Lines**: 97-99

**Current Code**:
```python
.with_standard_manager(
    chat_client=manager_client,
    max_round_count=self._max_rounds,  # Already uses config
    max_stall_count=3,
    max_reset_count=2,
)
```

**Issue**: Default `max_rounds` in `__init__` is 10, but workflow may need more for complex queries.

**Fix**: Verify the value flows through correctly. Add logging.

```python
logger.info(
    "Building Magentic workflow",
    max_rounds=self._max_rounds,
    max_stall=3,
    max_reset=2,
)
```

**Also check**: `src/orchestrator_factory.py` passes config correctly:
```python
return MagenticOrchestrator(
    max_rounds=config.max_iterations if config else 10,
)
```

---

### Phase 4: Fix Stale bioRxiv References

**Files to update**:

| File | Line | Change |
|------|------|--------|
| `src/orchestrator_magentic.py` | 131 | "bioRxiv" → "Europe PMC" |
| `src/agents/magentic_agents.py` | 32-33 | "bioRxiv" → "Europe PMC" |
| `src/app.py` | 202-203 | "bioRxiv" → "Europe PMC" |

**Search command to verify**:
```bash
grep -rn "bioRxiv\|biorxiv" src/
```

---

## Implementation Checklist

```
[ ] Phase 1: Write failing tests
    [ ] 1.1 Test ChatMessage text extraction
    [ ] 1.2 Test max rounds configuration
    [ ] 1.3 Test Europe PMC references

[ ] Phase 2: Fix ChatMessage extraction
    [ ] Update _process_event() in orchestrator_magentic.py
    [ ] Run test 1.1 - should pass

[ ] Phase 3: Fix max rounds
    [ ] Add logging to _build_workflow()
    [ ] Verify factory passes config correctly
    [ ] Run test 1.2 - should pass

[ ] Phase 4: Fix bioRxiv references
    [ ] Update orchestrator_magentic.py task prompt
    [ ] Update magentic_agents.py descriptions
    [ ] Update app.py UI text
    [ ] Run test 1.3 - should pass
    [ ] Run grep to verify no remaining refs

[ ] Final Verification
    [ ] make check passes
    [ ] All tests pass (108+)
    [ ] Manual test: run_magentic.py produces readable report
```

---

## Test Commands

```bash
# Run specific test file
uv run pytest tests/unit/test_orchestrator_magentic.py -v

# Run all tests
uv run pytest tests/unit/ -v

# Full check
make check

# Manual integration test
set -a && source .env && set +a
uv run python examples/orchestrator_demo/run_magentic.py "metformin alzheimer"
```

---

## Success Criteria

1. `run_magentic.py` outputs a readable research report (not `<ChatMessage object>`)
2. Report includes: Executive Summary, Key Findings, Drug Candidates, References
3. No "Max round count reached" error with default settings
4. No "bioRxiv" references anywhere in codebase
5. All 108+ tests pass
6. `make check` passes

---

## Files Modified

```
src/
├── orchestrator_magentic.py   # ChatMessage fix, logging
├── agents/magentic_agents.py  # bioRxiv → Europe PMC
└── app.py                     # bioRxiv → Europe PMC

tests/unit/
└── test_orchestrator_magentic.py  # NEW: 3 tests
```

---

## Notes for AI Agent

When implementing this fix plan:

1. **DO NOT** create mock data or fake responses
2. **DO** write real tests that verify actual behavior
3. **DO** run `make check` after each phase
4. **DO** test with real OpenAI API key via `.env`
5. **DO** preserve existing functionality - simple mode must still work
6. **DO NOT** over-engineer - minimal changes to fix the specific bugs
