# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

DeepBoner is an AI-native sexual health research agent. It uses a search-and-judge loop to autonomously search biomedical databases (PubMed, ClinicalTrials.gov, Europe PMC) and synthesize evidence for queries like "What drugs improve female libido post-menopause?" or "Evidence for testosterone therapy in women with HSDD?".

**Current Status:** Phases 1-13 COMPLETE (Foundation through Modal sandbox integration).

## Development Commands

```bash
# Install all dependencies (including dev)
make install   # or: uv sync --all-extras && uv run pre-commit install

# Run all quality checks (lint + typecheck + test) - MUST PASS BEFORE COMMIT
make check

# Individual commands
make test        # uv run pytest tests/unit/ -v
make lint        # uv run ruff check src tests
make format      # uv run ruff format src tests
make typecheck   # uv run mypy src
make test-cov    # uv run pytest --cov=src --cov-report=term-missing

# Run single test
uv run pytest tests/unit/utils/test_config.py::TestSettings::test_default_max_iterations -v

# Integration tests (real APIs)
uv run pytest -m integration
```

## Architecture

**Pattern**: Search-and-judge loop with multi-tool orchestration.

```text
User Question → Orchestrator
    ↓
Search Loop:
  1. Query PubMed, ClinicalTrials.gov, Europe PMC
  2. Gather evidence
  3. Judge quality ("Do we have enough?")
  4. If NO → Refine query, search more
  5. If YES → Synthesize findings (+ optional Modal analysis)
    ↓
Research Report with Citations
```

**Key Components**:

- `src/orchestrator.py` - Main agent loop
- `src/tools/pubmed.py` - PubMed E-utilities search
- `src/tools/clinicaltrials.py` - ClinicalTrials.gov API
- `src/tools/europepmc.py` - Europe PMC search
- `src/tools/code_execution.py` - Modal sandbox execution
- `src/tools/search_handler.py` - Scatter-gather orchestration
- `src/services/embeddings.py` - Semantic search & deduplication (ChromaDB)
- `src/services/statistical_analyzer.py` - Statistical analysis via Modal
- `src/agent_factory/judges.py` - LLM-based evidence assessment
- `src/agents/` - Magentic multi-agent mode (SearchAgent, JudgeAgent, etc.)
- `src/mcp_tools.py` - MCP tool wrappers for Claude Desktop
- `src/utils/config.py` - Pydantic Settings (loads from `.env`)
- `src/utils/models.py` - Evidence, Citation, SearchResult models
- `src/utils/exceptions.py` - Exception hierarchy
- `src/app.py` - Gradio UI with MCP server (HuggingFace Spaces)

**Break Conditions**: Judge approval, token budget (50K max), or max iterations (default 10).

## Configuration

Settings via pydantic-settings from `.env`:

- `LLM_PROVIDER`: "openai" or "anthropic"
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: LLM keys
- `NCBI_API_KEY`: Optional, for higher PubMed rate limits
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: For Modal sandbox (optional)
- `MAX_ITERATIONS`: 1-50, default 10
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR

## Exception Hierarchy

```text
DeepBonerError (base)
├── SearchError
│   └── RateLimitError
├── JudgeError
└── ConfigurationError
```

## Testing

- **TDD**: Write tests first in `tests/unit/`, implement in `src/`
- **Markers**: `unit`, `integration`, `slow`
- **Mocking**: `respx` for httpx, `pytest-mock` for general mocking
- **Fixtures**: `tests/conftest.py` has `mock_httpx_client`, `mock_llm_response`

## Coding Standards

- Python 3.11+, strict mypy, ruff (100-char lines)
- Type all functions, use Pydantic models for data
- Use `structlog` for logging, not print
- Conventional commits: `feat(scope):`, `fix:`, `docs:`

## Git Workflow

- `main`: Production-ready (GitHub)
- `dev`: Development integration (GitHub)
- Remote `origin`: GitHub (source of truth for PRs/code review)
- Remote `huggingface-upstream`: HuggingFace Spaces (deployment target)

**HuggingFace Spaces Collaboration:**

- Each contributor should use their own dev branch: `yourname-dev` (e.g., `vcms-dev`, `mario-dev`)
- **DO NOT push directly to `main` or `dev` on HuggingFace** - these can be overwritten easily
- GitHub is the source of truth; HuggingFace is for deployment/demo
- Consider using git hooks to prevent accidental pushes to protected branches
