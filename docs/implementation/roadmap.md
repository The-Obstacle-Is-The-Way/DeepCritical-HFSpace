# Implementation Roadmap: DeepCritical (Vertical Slices)

**Philosophy:** AI-Native Engineering, Vertical Slice Architecture, TDD, Modern Tooling (2025).

This roadmap defines the execution strategy to deliver **DeepCritical** effectively. We reject "overplanning" in favor of **ironclad, testable vertical slices**. Each phase delivers a fully functional slice of end-to-end value.

---

## The 2025 "Gucci" Tooling Stack

We are using the bleeding edge of Python engineering to ensure speed, safety, and developer joy.

| Category | Tool | Why? |
|----------|------|------|
| **Package Manager** | **`uv`** | Rust-based, 10-100x faster than pip/poetry. Manages python versions, venvs, and deps. |
| **Linting/Format** | **`ruff`** | Rust-based, instant. Replaces black, isort, flake8. |
| **Type Checking** | **`mypy`** | Strict static typing. Run via `uv run mypy`. |
| **Testing** | **`pytest`** | The standard. |
| **Test Plugins** | **`pytest-sugar`** | Instant feedback, progress bars. "Gucci" visuals. |
| **Test Plugins** | **`pytest-asyncio`** | Essential for our async agent loop. |
| **Test Plugins** | **`pytest-cov`** | Coverage reporting to ensure TDD adherence. |
| **Git Hooks** | **`pre-commit`** | Enforce ruff/mypy before commit. |

---

## Architecture: Vertical Slices

Instead of horizontal layers (e.g., "Building the Database Layer"), we build **Vertical Slices**.
Each slice implements a feature from **Entry Point (UI/API) -> Logic -> Data/External**.

### Directory Structure (Maintainer's Structure)

```bash
src/
├── app.py                      # Entry point (Gradio UI)
├── orchestrator.py             # Agent loop (Search -> Judge -> Loop)
├── agent_factory/              # Agent creation and judges
│   ├── __init__.py
│   ├── agents.py               # PydanticAI agent definitions
│   └── judges.py               # JudgeHandler for evidence assessment
├── tools/                      # Search tools
│   ├── __init__.py
│   ├── pubmed.py               # PubMed E-utilities tool
│   ├── websearch.py            # DuckDuckGo search tool
│   └── search_handler.py       # Orchestrates multiple tools
├── prompts/                    # Prompt templates
│   ├── __init__.py
│   └── judge.py                # Judge prompts
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── config.py               # Settings/configuration
│   ├── exceptions.py           # Custom exceptions
│   ├── models.py               # Shared Pydantic models
│   ├── dataloaders.py          # Data loading utilities
│   └── parsers.py              # Parsing utilities
├── middleware/                 # (Future: middleware components)
├── database_services/          # (Future: database integrations)
└── retrieval_factory/          # (Future: RAG components)

tests/
├── unit/
│   ├── tools/
│   │   ├── test_pubmed.py
│   │   ├── test_websearch.py
│   │   └── test_search_handler.py
│   ├── agent_factory/
│   │   └── test_judges.py
│   └── test_orchestrator.py
└── integration/
    └── test_pubmed_live.py
```

---

## Phased Execution Plan

### **Phase 1: Foundation & Tooling (Day 1)**
*Goal: A rock-solid, CI-ready environment with `uv` and `pytest` configured.*
- [ ] Initialize `pyproject.toml` with `uv`.
- [ ] Configure `ruff` (strict) and `mypy` (strict).
- [ ] Set up `pytest` with sugar and coverage.
- [ ] Implement `src/utils/config.py` (Configuration Slice).
- [ ] Implement `src/utils/exceptions.py` (Custom exceptions).
- **Deliverable**: A repo that passes CI with `uv run pytest`.

### **Phase 2: The "Search" Vertical Slice (Day 2)**
*Goal: Agent can receive a query and get raw results from PubMed/Web.*
- [ ] **TDD**: Write test for `SearchHandler`.
- [ ] Implement `src/tools/pubmed.py` (PubMed E-utilities).
- [ ] Implement `src/tools/websearch.py` (DuckDuckGo).
- [ ] Implement `src/tools/search_handler.py` (Orchestrates tools).
- [ ] Implement `src/utils/models.py` (Evidence, Citation, SearchResult).
- **Deliverable**: Function that takes "long covid" -> returns `List[Evidence]`.

### **Phase 3: The "Judge" Vertical Slice (Day 3)**
*Goal: Agent can decide if evidence is sufficient.*
- [ ] **TDD**: Write test for `JudgeHandler` (Mocked LLM).
- [ ] Implement `src/prompts/judge.py` (Structured outputs).
- [ ] Implement `src/agent_factory/judges.py` (LLM interaction).
- **Deliverable**: Function that takes `List[Evidence]` -> returns `JudgeAssessment`.

### **Phase 4: The "Loop" & UI Slice (Day 4)**
*Goal: End-to-End User Value.*
- [ ] Implement `src/orchestrator.py` (Connects Search + Judge loops).
- [ ] Build `src/app.py` (Gradio with Streaming).
- **Deliverable**: Working DeepCritical Agent on HuggingFace.

---

## Spec Documents

1. **[Phase 1 Spec: Foundation](01_phase_foundation.md)**
2. **[Phase 2 Spec: Search Slice](02_phase_search.md)**
3. **[Phase 3 Spec: Judge Slice](03_phase_judge.md)**
4. **[Phase 4 Spec: UI & Loop](04_phase_ui.md)**

*Start by reading Phase 1 Spec to initialize the repo.*
