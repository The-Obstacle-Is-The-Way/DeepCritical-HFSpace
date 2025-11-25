# Implementation Roadmap: DeepCritical (Vertical Slices)

**Philosophy:** AI-Native Engineering, Vertical Slice Architecture, TDD, Modern Tooling (2025).

This roadmap defines the execution strategy to deliver **DeepCritical** effectively. We reject "overplanning" in favor of **ironclad, testable vertical slices**. Each phase delivers a fully functional slice of end-to-end value.

**Total Estimated Effort**: 12-16 hours (can be done in 4 days)

---

## 🛠️ The 2025 "Gucci" Tooling Stack

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
| **Test Plugins** | **`pytest-mock`** | Easy mocking with `mocker` fixture. |
| **HTTP Mocking** | **`respx`** | Mock `httpx` requests in tests. |
| **Git Hooks** | **`pre-commit`** | Enforce ruff/mypy before commit. |
| **Retry Logic** | **`tenacity`** | Exponential backoff for API calls. |
| **Logging** | **`structlog`** | Structured JSON logging. |

---

## 🏗️ Architecture: Vertical Slices

Instead of horizontal layers (e.g., "Building the Database Layer"), we build **Vertical Slices**.
Each slice implements a feature from **Entry Point (UI/API) → Logic → Data/External**.

### Directory Structure (Maintainer's Template + Our Code)

We use the **existing scaffolding** from the maintainer, filling in the empty files.

```
deepcritical/
├── pyproject.toml          # All config in one file
├── .env.example            # Environment template
├── .pre-commit-config.yaml # Git hooks
├── Dockerfile              # Container build
├── README.md               # HuggingFace Space config
│
├── src/
│   ├── app.py              # Gradio entry point
│   ├── orchestrator.py     # Main agent loop (Search→Judge→Synthesize)
│   │
│   ├── agent_factory/      # Agent definitions
│   │   ├── __init__.py
│   │   ├── agents.py       # (Reserved for future agents)
│   │   └── judges.py       # JudgeHandler - LLM evidence assessment
│   │
│   ├── tools/              # Search tools
│   │   ├── __init__.py
│   │   ├── pubmed.py       # PubMedTool - NCBI E-utilities
│   │   ├── websearch.py    # WebTool - DuckDuckGo
│   │   └── search_handler.py # SearchHandler - orchestrates tools
│   │
│   ├── prompts/            # Prompt templates
│   │   ├── __init__.py
│   │   └── judge.py        # Judge system/user prompts
│   │
│   ├── utils/              # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py       # Settings via pydantic-settings
│   │   ├── exceptions.py   # Custom exceptions
│   │   └── models.py       # ALL Pydantic models (Evidence, JudgeAssessment, etc.)
│   │
│   ├── middleware/         # (Empty - reserved)
│   ├── database_services/  # (Empty - reserved)
│   └── retrieval_factory/  # (Empty - reserved)
│
└── tests/
    ├── __init__.py
    ├── conftest.py         # Shared fixtures
    │
    ├── unit/               # Fast, mocked tests
    │   ├── __init__.py
    │   ├── utils/          # Config, models tests
    │   ├── tools/          # PubMed, WebSearch tests
    │   └── agent_factory/  # Judge tests
    │
    └── integration/        # Real API tests (optional)
        └── __init__.py
```

---

## 🚀 Phased Execution Plan

### **Phase 1: Foundation & Tooling (~2-3 hours)**

*Goal: A rock-solid, CI-ready environment with `uv` and `pytest` configured.*

| Task | Output |
|------|--------|
| Install uv | `uv --version` works |
| Create pyproject.toml | All deps + config in one file |
| Set up directory structure | All `__init__.py` files created |
| Configure ruff + mypy | Strict settings |
| Create conftest.py | Shared pytest fixtures |
| Implement shared/config.py | Settings via pydantic-settings |
| Write first test | `test_config.py` passes |

**Deliverable**: `uv run pytest` passes with green output.

📄 **Spec Document**: [01_phase_foundation.md](01_phase_foundation.md)

---

### **Phase 2: The "Search" Vertical Slice (~3-4 hours)**

*Goal: Agent can receive a query and get raw results from PubMed/Web.*

| Task | Output |
|------|--------|
| Define Evidence/Citation models | Pydantic models |
| Implement PubMedTool | ESearch → EFetch → Evidence |
| Implement WebTool | DuckDuckGo → Evidence |
| Implement SearchHandler | Parallel search orchestration |
| Write unit tests | Mocked HTTP responses |

**Deliverable**: Function that takes "long covid" → returns `List[Evidence]`.

📄 **Spec Document**: [02_phase_search.md](02_phase_search.md)

---

### **Phase 3: The "Judge" Vertical Slice (~3-4 hours)**

*Goal: Agent can decide if evidence is sufficient.*

| Task | Output |
|------|--------|
| Define JudgeAssessment model | Structured output schema |
| Write prompt templates | System + user prompts |
| Implement JudgeHandler | PydanticAI agent with structured output |
| Write unit tests | Mocked LLM responses |

**Deliverable**: Function that takes `List[Evidence]` → returns `JudgeAssessment`.

📄 **Spec Document**: [03_phase_judge.md](03_phase_judge.md)

---

### **Phase 4: The "Orchestrator" & UI Slice (~4-5 hours)**

*Goal: End-to-End User Value.*

| Task | Output |
|------|--------|
| Define AgentEvent/State models | Event streaming types |
| Implement Orchestrator | Main while loop connecting Search→Judge |
| Implement report synthesis | Generate markdown report |
| Build Gradio UI | Streaming chat interface |
| Create Dockerfile | Container for deployment |
| Create HuggingFace README | Space configuration |
| Write unit tests | Mocked handlers |

**Deliverable**: Working DeepCritical Agent on localhost:7860.

📄 **Spec Document**: [04_phase_ui.md](04_phase_ui.md)

---

## 📜 Spec Documents Summary

| Phase | Document | Focus |
|-------|----------|-------|
| 1 | [01_phase_foundation.md](01_phase_foundation.md) | Tooling, config, TDD setup |
| 2 | [02_phase_search.md](02_phase_search.md) | PubMed + DuckDuckGo search |
| 3 | [03_phase_judge.md](03_phase_judge.md) | LLM evidence assessment |
| 4 | [04_phase_ui.md](04_phase_ui.md) | Orchestrator + Gradio + Deploy |

---

## ⚡ Quick Start Commands

```bash
# Phase 1: Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --name deepcritical
uv sync --all-extras
uv run pytest

# Phase 2-4: Development
uv run pytest tests/unit/ -v          # Run unit tests
uv run ruff check src tests           # Lint
uv run mypy src                       # Type check
uv run python src/app.py              # Run Gradio locally

# Deployment
docker build -t deepcritical .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... deepcritical
```

---

## 🎯 Definition of Done (MVP)

The MVP is **COMPLETE** when:

1. ✅ All unit tests pass (`uv run pytest`)
2. ✅ Ruff has no errors (`uv run ruff check`)
3. ✅ Mypy has no errors (`uv run mypy src`)
4. ✅ Gradio UI runs locally (`uv run python src/app.py`)
5. ✅ Can ask "Can metformin treat Alzheimer's?" and get a report
6. ✅ Report includes drug candidates, citations, and quality scores
7. ✅ Docker builds successfully
8. ✅ Deployable to HuggingFace Spaces

---

## 📊 Progress Tracker

| Phase | Status | Tests | Notes |
|-------|--------|-------|-------|
| 1: Foundation | ⬜ Pending | 0/5 | Start here |
| 2: Search | ⬜ Pending | 0/6 | Depends on Phase 1 |
| 3: Judge | ⬜ Pending | 0/5 | Depends on Phase 2 |
| 4: Orchestrator | ⬜ Pending | 0/4 | Depends on Phase 3 |

Update this table as you complete each phase!

---

*Start by reading [Phase 1 Spec](01_phase_foundation.md) to initialize the repo.*
