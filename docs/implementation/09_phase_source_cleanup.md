# Phase 9 Implementation Spec: Remove DuckDuckGo

**Goal**: Remove unreliable web search, focus on credible scientific sources.
**Philosophy**: "Scientific credibility over source quantity."
**Prerequisite**: Phase 8 complete (all agents working)
**Estimated Time**: 30-45 minutes

---

## 1. Why Remove DuckDuckGo?

### Current Problems

| Issue | Impact |
|-------|--------|
| Rate-limited aggressively | Returns 0 results frequently |
| Not peer-reviewed | Random blogs, news, misinformation |
| Not citable | Cannot use in scientific reports |
| Adds noise | Dilutes quality evidence |

### After Removal

| Benefit | Impact |
|---------|--------|
| Cleaner codebase | -150 lines of dead code |
| No rate limit failures | 100% source reliability |
| Scientific credibility | All sources peer-reviewed/preprint |
| Simpler debugging | Fewer failure modes |

---

## 2. Files to Modify/Delete

### 2.1 DELETE: `src/tools/websearch.py`

```bash
# File to delete entirely
src/tools/websearch.py  # ~80 lines
```

### 2.2 MODIFY: SearchHandler Usage

Update all files that instantiate `SearchHandler` with `WebTool()`:

| File | Change |
|------|--------|
| `examples/search_demo/run_search.py` | Remove `WebTool()` from tools list |
| `examples/hypothesis_demo/run_hypothesis.py` | Remove `WebTool()` from tools list |
| `examples/full_stack_demo/run_full.py` | Remove `WebTool()` from tools list |
| `examples/orchestrator_demo/run_agent.py` | Remove `WebTool()` from tools list |
| `examples/orchestrator_demo/run_magentic.py` | Remove `WebTool()` from tools list |

### 2.3 MODIFY: Type Definitions

Update `src/utils/models.py`:

```python
# BEFORE
sources_searched: list[Literal["pubmed", "web"]]

# AFTER (Phase 9)
sources_searched: list[Literal["pubmed"]]

# AFTER (Phase 10-11)
sources_searched: list[Literal["pubmed", "clinicaltrials", "biorxiv"]]
```

### 2.4 DELETE: Tests for WebTool

```bash
# File to delete
tests/unit/tools/test_websearch.py
```

---

## 3. TDD Implementation

### 3.1 Test: SearchHandler Works Without WebTool

```python
# tests/unit/tools/test_search_handler.py

@pytest.mark.asyncio
async def test_search_handler_pubmed_only():
    """SearchHandler should work with only PubMed tool."""
    from src.tools.pubmed import PubMedTool
    from src.tools.search_handler import SearchHandler

    handler = SearchHandler(tools=[PubMedTool()], timeout=30.0)

    # Should not raise
    result = await handler.execute("metformin diabetes", max_results_per_tool=3)

    assert result.sources_searched == ["pubmed"]
    assert "web" not in result.sources_searched
    assert len(result.errors) == 0  # No failures
```

### 3.2 Test: WebTool Import Fails (Deleted)

```python
# tests/unit/tools/test_websearch_removed.py

def test_websearch_module_deleted():
    """WebTool should no longer exist."""
    with pytest.raises(ImportError):
        from src.tools.websearch import WebTool
```

### 3.3 Test: Examples Don't Reference WebTool

```python
# tests/unit/test_no_webtool_references.py

import ast
import pathlib

def test_examples_no_webtool_imports():
    """No example files should import WebTool."""
    examples_dir = pathlib.Path("examples")

    for py_file in examples_dir.rglob("*.py"):
        content = py_file.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "websearch" in node.module:
                    pytest.fail(f"{py_file} imports websearch (should be removed)")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "websearch" in alias.name:
                        pytest.fail(f"{py_file} imports websearch (should be removed)")
```

---

## 4. Step-by-Step Implementation

### Step 1: Write Tests First (TDD)

```bash
# Create the test file
touch tests/unit/tools/test_websearch_removed.py
# Write the tests from section 3
```

### Step 2: Run Tests (Should Fail)

```bash
uv run pytest tests/unit/tools/test_websearch_removed.py -v
# Expected: FAIL (websearch still exists)
```

### Step 3: Delete WebTool

```bash
rm src/tools/websearch.py
rm tests/unit/tools/test_websearch.py
```

### Step 4: Update SearchHandler Usages

```python
# BEFORE (in each example file)
from src.tools.websearch import WebTool
search_handler = SearchHandler(tools=[PubMedTool(), WebTool()], timeout=30.0)

# AFTER
from src.tools.pubmed import PubMedTool
search_handler = SearchHandler(tools=[PubMedTool()], timeout=30.0)
```

### Step 5: Update Type Definitions

```python
# src/utils/models.py
# BEFORE
sources_searched: list[Literal["pubmed", "web"]]

# AFTER
sources_searched: list[Literal["pubmed"]]
```

### Step 6: Run All Tests

```bash
uv run pytest tests/unit/ -v
# Expected: ALL PASS
```

### Step 7: Run Lints

```bash
uv run ruff check src tests examples
uv run mypy src
# Expected: No errors
```

---

## 5. Definition of Done

Phase 9 is **COMPLETE** when:

- [ ] `src/tools/websearch.py` deleted
- [ ] `tests/unit/tools/test_websearch.py` deleted
- [ ] All example files updated (no WebTool imports)
- [ ] Type definitions updated in models.py
- [ ] New tests verify WebTool is removed
- [ ] All existing tests pass
- [ ] Lints pass
- [ ] Examples run successfully with PubMed only

---

## 6. Verification Commands

```bash
# 1. Verify websearch.py is gone
ls src/tools/websearch.py 2>&1 | grep "No such file"

# 2. Verify no WebTool imports remain
grep -r "WebTool" src/ examples/ && echo "FAIL: WebTool references found" || echo "PASS"
grep -r "websearch" src/ examples/ && echo "FAIL: websearch references found" || echo "PASS"

# 3. Run tests
uv run pytest tests/unit/ -v

# 4. Run example (should work)
source .env && uv run python examples/search_demo/run_search.py "metformin cancer"
```

---

## 7. Rollback Plan

If something breaks:

```bash
git checkout HEAD -- src/tools/websearch.py
git checkout HEAD -- tests/unit/tools/test_websearch.py
```

---

## 8. Value Delivered

| Before | After |
|--------|-------|
| 2 search sources (1 broken) | 1 reliable source |
| Rate limit failures | No failures |
| Web noise in results | Pure scientific sources |
| ~230 lines for websearch | 0 lines |

**Net effect**: Simpler, more reliable, more credible.
