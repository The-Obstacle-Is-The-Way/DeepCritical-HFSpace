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
