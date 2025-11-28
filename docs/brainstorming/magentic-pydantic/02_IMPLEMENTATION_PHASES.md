# Implementation Phases: Dual-Mode Agent System

**Date:** November 27, 2025
**Status:** IMPLEMENTATION PLAN (REVISED)
**Strategy:** TDD (Test-Driven Development), SOLID Principles
**Dependency Strategy:** PyPI (agent-framework-core)

---

## Phase 0: Environment Validation & Cleanup

**Goal:** Ensure clean state and dependencies are correctly installed.

### Step 0.1: Verify PyPI Package
The `agent-framework-core` package is published on PyPI by Microsoft. Verify installation:

```bash
uv sync --all-extras
python -c "from agent_framework import ChatAgent; print('OK')"
```

### Step 0.2: Branch State
We are on `feat/dual-mode-architecture`. Ensure it is up to date with `origin/dev` before starting.

**Note:** The `reference_repos/agent-framework` folder is kept for reference/documentation only.
The production dependency uses the official PyPI release.

---

## Phase 1: Pydantic-AI Improvements (Simple Mode)

**Goal:** Implement `HuggingFaceModel` support in `JudgeHandler` using strict TDD.

### Step 1.1: Test First (Red)
Create `tests/unit/agent_factory/test_judges_factory.py`:
- Test `get_model()` returns `HuggingFaceModel` when `LLM_PROVIDER=huggingface`.
- Test `get_model()` respects `HF_TOKEN`.
- Test fallback to OpenAI.

### Step 1.2: Implementation (Green)
Update `src/utils/config.py`:
- Add `huggingface_model` and `hf_token` fields.

Update `src/agent_factory/judges.py`:
- Implement `get_model` with the logic derived from the tests.
- Use dependency injection for the model where possible.

### Step 1.3: Refactor
Ensure `JudgeHandler` is loosely coupled from the specific model provider.

---

## Phase 2: Orchestrator Factory (The Switch)

**Goal:** Implement the factory pattern to switch between Simple and Advanced modes.

### Step 2.1: Test First (Red)
Create `tests/unit/test_orchestrator_factory.py`:
- Test `create_orchestrator` returns `Orchestrator` (simple) when API keys are missing.
- Test `create_orchestrator` returns `MagenticOrchestrator` (advanced) when OpenAI key exists.
- Test explicit mode override.

### Step 2.2: Implementation (Green)
Update `src/orchestrator_factory.py` to implement the selection logic.

---

## Phase 3: Agent Framework Integration (Advanced Mode)

**Goal:** Integrate Microsoft Agent Framework from PyPI.

### Step 3.1: Dependency Management
The `agent-framework-core` package is installed from PyPI:
```toml
[project.optional-dependencies]
magentic = [
    "agent-framework-core>=1.0.0b251120,<2.0.0",  # Microsoft Agent Framework (PyPI)
]
```
Install with: `uv sync --all-extras`

### Step 3.2: Verify Imports (Test First)
Create `tests/unit/agents/test_agent_imports.py`:
- Verify `from agent_framework import ChatAgent` works.
- Verify instantiation of `ChatAgent` with a mock client.

### Step 3.3: Update Agents
Refactor `src/agents/*.py` to ensure they match the exact signature of the local `ChatAgent` class.
- **SOLID:** Ensure agents have single responsibilities.
- **DRY:** Share tool definitions between Pydantic-AI simple mode and Agent Framework advanced mode.

---

## Phase 4: UI & End-to-End Verification

**Goal:** Update Gradio to reflect the active mode.

### Step 4.1: UI Updates
Update `src/app.py` to display "Simple Mode" vs "Advanced Mode".

### Step 4.2: End-to-End Test
Run the full loop:
1. Simple Mode (No Keys) -> Search -> Judge (HF) -> Report.
2. Advanced Mode (OpenAI Key) -> SearchAgent -> JudgeAgent -> ReportAgent.

---

## Phase 5: Cleanup & Documentation

- Remove unused code.
- Update main README.md.
- Final `make check`.