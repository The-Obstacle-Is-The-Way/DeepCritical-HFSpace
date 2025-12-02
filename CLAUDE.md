# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepBoner is an AI-native sexual health research agent. It uses a search-and-judge loop to autonomously search biomedical databases (PubMed, ClinicalTrials.gov, Europe PMC) and synthesize evidence for queries like "What drugs improve female libido post-menopause?" or "Evidence for testosterone therapy in women with HSDD?".

**Current Status:** Phases 1-14 COMPLETE (Foundation through Demo Submission).

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

- `src/orchestrators/` - Unified orchestrator package
  - `advanced.py` - Main orchestrator (handles both Free and Paid tiers)
  - `factory.py` - Auto-selects backend based on API key presence
  - `langgraph_orchestrator.py` - LangGraph-based workflow (experimental)
- `src/clients/` - LLM backend adapters
  - `factory.py` - Auto-selects: OpenAI (if key) or HuggingFace (free)
  - `huggingface.py` - HuggingFace adapter for free tier
- `src/tools/pubmed.py` - PubMed E-utilities search
- `src/tools/clinicaltrials.py` - ClinicalTrials.gov API
- `src/tools/europepmc.py` - Europe PMC search
- `src/tools/code_execution.py` - Modal sandbox execution
- `src/tools/search_handler.py` - Scatter-gather orchestration
- `src/services/embeddings.py` - Local embeddings (sentence-transformers, in-memory)
- `src/services/llamaindex_rag.py` - Premium embeddings (OpenAI, persistent ChromaDB)
- `src/services/embedding_protocol.py` - Protocol interface for embedding services
- `src/services/research_memory.py` - Shared memory layer for research state
- `src/services/statistical_analyzer.py` - Statistical analysis via Modal
- `src/utils/service_loader.py` - Tiered service selection (free vs premium)
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
├── ConfigurationError
└── EmbeddingError
```

## Testing

- **TDD**: Write tests first in `tests/unit/`, implement in `src/`
- **Markers**: `unit`, `integration`, `slow`
- **Mocking**: `respx` for httpx, `pytest-mock` for general mocking
- **Fixtures**: `tests/conftest.py` has `mock_httpx_client`, `mock_llm_response`

## LLM Model Defaults (November 2025)

Given the rapid advancements, as of November 29, 2025, the DeepBoner project uses the following default LLM models in its configuration (`src/utils/config.py`):

- **OpenAI:** `gpt-5`
  - Current flagship model (November 2025). Requires Tier 5 access.
- **Anthropic:** `claude-sonnet-4-5-20250929`
  - This is the mid-range Claude 4.5 model, released on September 29, 2025.
  - The flagship `Claude Opus 4.5` (released November 24, 2025) is also available and can be configured by advanced users for enhanced capabilities.
- **HuggingFace (Free Tier):** `Qwen/Qwen2.5-72B-Instruct`
  - Changed from Llama-3.1-70B (Dec 2025) due to HuggingFace routing Llama to Hyperbolic provider which has unreliable "staging mode" auth.
  - Qwen 2.5 72B offers comparable quality and works reliably via HuggingFace's native infrastructure.

It is crucial to keep these defaults updated as the LLM landscape evolves.

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
