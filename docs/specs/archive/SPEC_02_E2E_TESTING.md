# SPEC 02: End-to-End Testing

## Priority: P1 (Validation Before Features)

## Problem Statement

We have 141 unit tests that verify individual components work, but **no test that proves the full pipeline produces useful research output**.

We don't know if:
1. Simple mode produces a valid report
2. Advanced mode produces a valid report
3. The output is actually useful (has citations, mechanisms, etc.)

**Golden Rule**: Don't add features (OpenAlex, persistence) until we prove current features work.

## What We Need to Test

### Level 1: Smoke Test (Does it run?)

```python
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_simple_mode_completes(mock_search_handler, mock_judge_handler):
    """Verify Simple mode runs without crashing."""
    from src.orchestrator import Orchestrator
    from src.utils.models import OrchestratorConfig

    config = OrchestratorConfig(max_iterations=2)
    orchestrator = Orchestrator(
        search_handler=mock_search_handler,
        judge_handler=mock_judge_handler,
        config=config,
        enable_analysis=False,
        enable_embeddings=False,
    )

    events = []
    async for event in orchestrator.run("test query"):
        events.append(event)

    # Must complete
    assert any(e.type == "complete" for e in events)
    # Must not error
    assert not any(e.type == "error" for e in events)
```

### Level 2: Structure Test (Is output valid?)

```python
@pytest.mark.e2e
async def test_output_has_required_fields():
    """Verify output contains expected structure."""
    result = await run_research("metformin for PCOS")

    # Must have citations
    assert len(result.citations) >= 1

    # Must have some text
    assert len(result.report) > 100

    # Must mention the query topic
    assert "metformin" in result.report.lower() or "pcos" in result.report.lower()
```

### Level 3: Quality Test (Is output useful?)

```python
@pytest.mark.e2e
async def test_output_quality():
    """Verify output contains actionable research."""
    result = await run_research("drugs for female libido")

    # Should have PMIDs or NCT IDs
    has_citations = any(
        "PMID" in str(c) or "NCT" in str(c)
        for c in result.citations
    )
    assert has_citations, "No real citations found"

    # Should discuss mechanism
    mechanism_words = ["mechanism", "pathway", "receptor", "target"]
    has_mechanism = any(w in result.report.lower() for w in mechanism_words)
    assert has_mechanism, "No mechanism discussion found"
```

## Test Strategy

### Mocking Strategy

For CI/fast tests, mock external APIs via pytest fixtures in `tests/e2e/conftest.py`:

```python
@pytest.fixture
def mock_search_handler():
    """Return a mock search handler that returns fake evidence."""
    from unittest.mock import MagicMock
    from src.utils.models import Citation, Evidence, SearchResult

    async def mock_execute(query: str):
        return SearchResult(
            evidence=[
                Evidence(
                    content="Study on test query showing positive results...",
                    citation=Citation(
                        source="pubmed",
                        title="Study on test query",
                        url="https://pubmed.example.com/123",
                        date="2024",
                    ),
                )
            ],
            sources_searched=["pubmed", "clinicaltrials"],
        )

    mock = MagicMock()
    mock.execute = mock_execute
    return mock

@pytest.fixture
def mock_judge_handler():
    """Return a mock judge that always says 'synthesize'."""
    from unittest.mock import MagicMock
    from src.utils.models import JudgeAssessment

    async def mock_assess(evidence, query):
        return JudgeAssessment(
            sufficient=True,
            reasoning="Mock: Evidence is sufficient",
            suggested_refinements=[],
            key_findings=["Finding 1", "Finding 2"],
            evidence_gaps=[],
            recommended_drugs=["MockDrug A", "MockDrug B"],
        )

    mock = MagicMock()
    mock.assess = mock_assess
    return mock
```

### Integration Tests (Real APIs)

For validation, run against real APIs (marked `@pytest.mark.integration`):

```python
@pytest.mark.integration
@pytest.mark.slow
async def test_real_pubmed_search():
    """Integration test with real PubMed API."""
    # Requires NCBI_API_KEY in env
    ...
```

## Test Matrix

| Mode | Mock | Real API | Status |
|------|------|----------|--------|
| Simple (Free) | ✅ Done | ⏳ Optional | ✅ IMPLEMENTED |
| Advanced (OpenAI) | ✅ Done | ⏳ Optional | ✅ IMPLEMENTED |

## Directory Structure

```
tests/
├── unit/           # Existing 141 tests
├── integration/    # Real API tests (existing)
└── e2e/            # NEW: Full pipeline tests
    ├── conftest.py         # E2E fixtures
    ├── test_simple_mode.py # Simple mode E2E
    └── test_advanced_mode.py # Magentic mode E2E
```

## Acceptance Criteria

- [x] E2E test for Simple mode (mocked)
- [x] E2E test for Advanced mode (mocked)
- [x] Tests validate output structure
- [x] Tests run in CI (<2 minutes)
- [ ] At least one integration test with real API (existing in tests/integration/)

**Status: IMPLEMENTED** (commit b1d094d)

## Why Before OpenAlex?

1. **Prove current system works** before adding complexity
2. **Establish baseline** - what does "good output" look like?
3. **Catch regressions** - future changes won't break core functionality
4. **Confidence for hackathon** - we know the demo will produce something

## Related Issues

- #47: E2E Testing - Does Pipeline Actually Generate Useful Reports?
- #65: Demo timing (must fix first to make E2E tests practical)

## Files Created

1. `tests/e2e/conftest.py` - E2E fixtures (mock_search_handler, mock_judge_handler)
2. `tests/e2e/test_simple_mode.py` - Simple mode tests (2 tests)
3. `tests/e2e/test_advanced_mode.py` - Advanced mode tests (1 test, mocked workflow)
