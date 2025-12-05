# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

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
- `src/tools/search_handler.py` - Scatter-gather orchestration
- `src/services/embeddings.py` - Local embeddings (sentence-transformers, in-memory)
- `src/services/llamaindex_rag.py` - Premium embeddings (OpenAI, persistent ChromaDB)
- `src/services/embedding_protocol.py` - Protocol interface for embedding services
- `src/services/research_memory.py` - Shared memory layer for research state
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

- `LLM_PROVIDER`: "openai" or "huggingface"
- `OPENAI_API_KEY`: LLM keys
- `NCBI_API_KEY`: Optional, for higher PubMed rate limits
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

## LLM Model Defaults (December 2025)

Default models in `src/utils/config.py`:

- **OpenAI:** `gpt-5` - Flagship model
- **HuggingFace (Free Tier):** `Qwen/Qwen2.5-7B-Instruct` - See critical note below

---

## ⚠️ OpenAI API Keys

**If you have a valid OpenAI API key, it will work. Period.**

- BYOK (Bring Your Own Key) auto-detects `sk-...` prefix and routes to OpenAI
- If you get errors, the key is **invalid or expired** - NOT an access tier issue
- **NEVER suggest "access tier" or "upgrade your plan"** - this is not how OpenAI works for API keys
- Valid keys work. Invalid keys don't. That's it.

---

## ⚠️ CRITICAL: HuggingFace Free Tier Architecture

**THIS IS IMPORTANT - READ BEFORE CHANGING THE FREE TIER MODEL**

HuggingFace has TWO execution paths for inference:

| Path | Host | Reliability | Model Size |
|------|------|-------------|------------|
| **Native Serverless** | HuggingFace infrastructure | ✅ High | < 30B params |
| **Inference Providers** | Third-party (Novita, Hyperbolic) | ❌ Unreliable | 70B+ params |

**The Trap:** When you request a large model (70B+) without a paid API key, HuggingFace **silently routes** the request to third-party providers. These providers have:
- 500 Internal Server Errors (Novita - current)
- 401 "Staging Mode" auth failures (Hyperbolic - past)

**The Rule:** Free Tier MUST use models < 30B to stay on native infrastructure.

**Current Safe Models (Dec 2025):**
| Model | Size | Status |
|-------|------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ✅ **DEFAULT** - Native, reliable |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | ✅ Native, reliable |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | ❌ Routed to Novita (500 errors) |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | ❌ Routed to Hyperbolic (401 errors) |

**See:** `HF_FREE_TIER_ANALYSIS.md` for full analysis.

---

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
