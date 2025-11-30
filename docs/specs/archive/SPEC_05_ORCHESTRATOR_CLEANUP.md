# SPEC 05: Orchestrator Module Cleanup

## Priority: P3 (Code Hygiene)

## Problem Statement

The codebase has an inconsistent orchestrator organization:

```
src/
├── orchestrator/              # EMPTY folder (just . and ..)
├── orchestrator.py            # Simple mode (15KB, 67% coverage)
├── orchestrator_factory.py    # Factory pattern (2.5KB, 87% coverage)
├── orchestrator_hierarchical.py  # Unused (3KB, 0% coverage)
└── orchestrator_magentic.py   # Advanced mode (11KB, 68% coverage)
```

## Related Issues

- GitHub Issue #67: Clean up empty src/orchestrator/ folder

## Analysis

### Empty Folder
The `src/orchestrator/` folder was created but never populated. All orchestrator implementations remain flat in `src/`.

### Dead Code
`orchestrator_hierarchical.py` has **0% test coverage** and appears to be an early prototype that was never integrated:
- Not imported anywhere in production code
- Not referenced in any tests
- Pattern doesn't match current architecture

### Import Pattern
All 30+ imports use the flat structure:
```python
from src.orchestrator import Orchestrator
from src.orchestrator_factory import create_orchestrator
from src.orchestrator_magentic import MagenticOrchestrator
```

## Options

### Option A: Minimal Cleanup (Recommended)

Delete the empty folder and dead code:

```bash
rm -rf src/orchestrator/
rm src/orchestrator_hierarchical.py
```

**Pros**: Zero import changes, minimal risk, quick
**Cons**: Flat structure remains

### Option B: Full Consolidation (Future)

Move everything into a proper module:

```
src/orchestrator/
├── __init__.py        # Re-export for backwards compat
├── base.py            # Shared protocols/types
├── simple.py          # From orchestrator.py
├── magentic.py        # From orchestrator_magentic.py
└── factory.py         # From orchestrator_factory.py
```

**Pros**: Cleaner organization, better separation
**Cons**: 30+ import changes, risk of breakage, time investment

### Option C: Hybrid (Pragmatic)

Delete empty folder + dead code now. Create `src/orchestrator/__init__.py` that re-exports from flat files:

```python
# src/orchestrator/__init__.py
from src.orchestrator import Orchestrator
from src.orchestrator_factory import create_orchestrator
from src.orchestrator_magentic import MagenticOrchestrator

__all__ = ["Orchestrator", "create_orchestrator", "MagenticOrchestrator"]
```

**Problem**: This creates confusing import semantics (`src.orchestrator` would be both a module and a file).

## Recommendation

**Option A** for now. The flat structure works fine and changing it provides no functional benefit. The empty folder and dead code should be removed.

Option B can be revisited post-hackathon when there's time for a proper refactor.

## Implementation

### Step 1: Remove Empty Folder

```bash
rm -rf src/orchestrator/
```

### Step 2: Remove Dead Code (Optional)

```bash
rm src/orchestrator_hierarchical.py
```

If keeping for reference, add a deprecation notice:
```python
# src/orchestrator_hierarchical.py
"""
DEPRECATED: Unused hierarchical orchestrator prototype.
Kept for reference only. See orchestrator.py (simple) or
orchestrator_magentic.py (advanced) for active implementations.
"""
```

### Step 3: Verify

```bash
make check  # All 142 tests should pass
```

## Acceptance Criteria

- [x] Empty `src/orchestrator/` folder deleted
- [x] No broken imports (grep for `from src.orchestrator/`)
- [x] Tests pass (154 unit tests)
- [x] `orchestrator_hierarchical.py` removed

**Status: IMPLEMENTED** (commit cb46aac)

## Files to Modify

1. `src/orchestrator/` - DELETE (empty folder)
2. `src/orchestrator_hierarchical.py` - DELETE or add deprecation notice

## Test Plan

```bash
# Verify nothing imports from the folder path
grep -r "from src.orchestrator/" src tests
# Should return nothing

# Verify nothing imports hierarchical
grep -r "orchestrator_hierarchical" src tests
# Should return nothing (except possibly this spec)

# Run full test suite
make check
```

## Risk Assessment

| Action | Risk | Mitigation |
|--------|------|------------|
| Delete empty folder | None | It's empty, nothing uses it |
| Delete hierarchical.py | Low | 0% coverage, no imports |
| Full consolidation | Medium | Many import changes |

## Time Estimate

- Option A: 5 minutes
- Option B: 1-2 hours (plus testing)
