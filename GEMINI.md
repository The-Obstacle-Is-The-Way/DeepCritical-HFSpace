# DeepBoner Context

## Project Overview

**DeepBoner** is an AI-native Sexual Health Research Agent.
**Goal:** To accelerate research into sexual health, wellness, and reproductive medicine by intelligently searching biomedical literature (PubMed, ClinicalTrials.gov, Europe PMC), evaluating evidence, and synthesizing findings.

**Architecture:**
The project follows a **Vertical Slice Architecture** (Search -> Judge -> Orchestrator) and adheres to **Strict TDD** (Test-Driven Development).

**Current Status:** Phases 1-14 COMPLETE (Foundation through Demo Submission).

## Tech Stack & Tooling

- **Language:** Python 3.11 (Pinned)
- **Package Manager:** `uv` (Rust-based, extremely fast)
- **Frameworks:** `pydantic`, `pydantic-ai`, `httpx`, `gradio[mcp]`
- **Vector DB:** `chromadb` with `sentence-transformers` for semantic search
- **Code Execution:** `modal` for secure sandboxed Python execution
- **Testing:** `pytest`, `pytest-asyncio`, `respx` (for mocking)
- **Quality:** `ruff` (linting/formatting), `mypy` (strict type checking), `pre-commit`

## Building & Running

| Command | Description |
| :--- | :--- |
| `make install` | Install dependencies and pre-commit hooks. |
| `make test` | Run unit tests. |
| `make lint` | Run Ruff linter. |
| `make format` | Run Ruff formatter. |
| `make typecheck` | Run Mypy static type checker. |
| `make check` | **The Golden Gate:** Runs lint, typecheck, and test. Must pass before committing. |
| `make clean` | Clean up cache and artifacts. |

## Directory Structure

- `src/`: Source code
  - `utils/`: Shared utilities (`config.py`, `exceptions.py`, `models.py`)
  - `tools/`: Search tools (`pubmed.py`, `clinicaltrials.py`, `europepmc.py`, `code_execution.py`)
  - `services/`: Services (`embeddings.py`, `statistical_analyzer.py`)
  - `agents/`: Magentic multi-agent mode agents
  - `agent_factory/`: Agent definitions (judges, prompts)
  - `mcp_tools.py`: MCP tool wrappers for Claude Desktop integration
  - `app.py`: Gradio UI with MCP server
- `tests/`: Test suite
  - `unit/`: Isolated unit tests (Mocked)
  - `integration/`: Real API tests (Marked as slow/integration)
- `docs/`: Documentation and Implementation Specs
- `examples/`: Working demos for each phase

## Key Components

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
- `src/mcp_tools.py` - MCP tool wrappers
- `src/app.py` - Gradio UI (HuggingFace Spaces) with MCP server

## Configuration

Settings via pydantic-settings from `.env`:

- `LLM_PROVIDER`: "openai" or "anthropic"
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: LLM keys
- `NCBI_API_KEY`: Optional, for higher PubMed rate limits
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: For Modal sandbox (optional)
- `MAX_ITERATIONS`: 1-50, default 10
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR

## LLM Model Defaults (December 2025)

Default models in `src/utils/config.py`:

- **OpenAI:** `gpt-5` - Flagship model
- **HuggingFace (Free Tier):** `Qwen/Qwen2.5-7B-Instruct` - See critical note below

**NOTE:** Anthropic is NOT supported (no embeddings API). See `P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md`.

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

## Development Conventions

1. **Strict TDD:** Write failing tests in `tests/unit/` *before* implementing logic in `src/`.
2. **Type Safety:** All code must pass `mypy --strict`. Use Pydantic models for data exchange.
3. **Linting:** Zero tolerance for Ruff errors.
4. **Mocking:** Use `respx` or `unittest.mock` for all external API calls in unit tests.
5. **Vertical Slices:** Implement features end-to-end rather than layer-by-layer.

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
