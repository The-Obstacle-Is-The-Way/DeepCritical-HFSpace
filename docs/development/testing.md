# Testing Strategy
## ensuring DeepBoner is Ironclad

---

## Overview

Our testing strategy follows a strict **Pyramid of Reliability**:
1. **Unit Tests**: Fast, isolated logic checks (60% of tests)
2. **Integration Tests**: Tool interactions & Agent loops (30% of tests)
3. **E2E / Regression Tests**: Full research workflows (10% of tests)

**Goal**: Ship a research agent that doesn't hallucinate, crash on API limits, or burn $100 in tokens by accident.

---

## 1. Unit Tests (Fast & Cheap)

**Location**: `tests/unit/`

Focus on individual components without external network calls. Mock everything.

### Key Test Cases

#### Agent Logic
- **Initialization**: Verify default config loads correctly.
- **State Updates**: Ensure `ResearchState` updates correctly (e.g., token counts increment).
- **Budget Checks**: Test `should_continue()` returns `False` when budget exceeded.
- **Error Handling**: Test partial failure recovery (e.g., one tool fails, agent continues).

#### Tools (Mocked)
- **Parser Logic**: Feed raw XML/JSON to tool parsers and verify `Evidence` objects.
- **Validation**: Ensure tools reject invalid queries (empty strings, etc.).

#### Judge Prompts
- **Schema Compliance**: Verify prompt templates generate valid JSON structure instructions.
- **Variable Injection**: Ensure `{question}` and `{context}` are injected correctly into prompts.

```python
# Example: Testing State Logic
def test_budget_stop():
    state = ResearchState(tokens_used=50001, max_tokens=50000)
    assert should_continue(state) is False
```

---

## 2. Integration Tests (Realistic & Mocked I/O)

**Location**: `tests/integration/`

Focus on the interaction between the Orchestrator, Tools, and LLM Judge. Use **VCR.py** or **Replay** patterns to record/replay API calls to save money/time.

### Key Test Cases

#### Search Loop
- **Iteration Flow**: Verify agent performs Search -> Judge -> Search loop.
- **Tool Selection**: Verify correct tools are called based on judge output (mocked judge).
- **Context Accumulation**: Ensure findings from Iteration 1 are passed to Iteration 2.

#### MCP Server Integration
- **Server Startup**: Verify MCP server starts and exposes tools.
- **Client Connection**: Verify agent can call tools via MCP protocol.

```python
# Example: Testing Search Loop with Mocked Tools
async def test_search_loop_flow():
    agent = ResearchAgent(tools=[MockPubMed(), MockWeb()])
    report = await agent.run("test query")
    assert agent.state.iterations > 0
    assert len(report.sources) > 0
```

---

## 3. End-to-End (E2E) Tests (The "Real Deal")

**Location**: `tests/e2e/`

Run against **real APIs** (with strict rate limits) to verify system integrity. Run these **on demand** or **nightly**, not on every commit.

### Key Test Cases

#### The "Golden Query"
Run the primary demo query: *"What existing drugs might help treat long COVID fatigue?"*
- **Success Criteria**:
  - Returns at least 2 valid drug candidates (e.g., CoQ10, LDN).
  - Includes citations from PubMed.
  - Completes within 3 iterations.
  - JSON output matches schema.

#### Deployment Smoke Test
- **Gradio UI**: Verify UI launches and accepts input.
- **Streaming**: Verify generator yields chunks (first chunk within 2s).

---

## 4. Tools & Config

### Pytest Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: fast, isolated tests",
    "integration: mocked network tests",
    "e2e: real network tests (slow, expensive)"
]
asyncio_mode = "auto"
```

### CI/CD Pipeline (GitHub Actions)
1. **Lint**: `ruff check .`
2. **Type Check**: `mypy .`
3. **Unit**: `pytest -m unit`
4. **Integration**: `pytest -m integration`
5. **E2E**: (Manual trigger only)

---

## 5. Anti-Hallucination Validation

How do we test if the agent is lying?

1. **Citation Check**:
   - Regex verify that every `[PMID: 12345]` in the report exists in the `Evidence` list.
   - Fail if a citation is "orphaned" (hallucinated ID).

2. **Negative Constraints**:
   - Test queries for fake diseases ("Ligma syndrome") -> Agent should return "No evidence found".

---

## Checklist for Implementation

- [ ] Set up `tests/` directory structure
- [ ] Configure `pytest` and `vcrpy`
- [ ] Create `tests/fixtures/` for mock data (PubMed XMLs)
- [ ] Write first unit test for `ResearchState`
