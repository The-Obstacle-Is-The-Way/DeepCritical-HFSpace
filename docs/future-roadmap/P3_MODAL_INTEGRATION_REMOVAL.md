# P3 Tech Debt: Modal Integration Removal

**Date**: 2025-12-04
**Status**: DONE
**Severity**: P3 (Tech Debt - Not blocking functionality)
**Component**: Multiple files

---

## Executive Summary

Modal (cloud function execution platform) is integrated throughout the codebase but was decided against for this project. This creates potential confusion and dead code paths that should be cleaned up when time permits.

---

## Affected Files

The following files contain Modal references:

| File | Usage |
|------|-------|
| `src/utils/config.py` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` settings |
| `src/utils/service_loader.py` | Modal service initialization |
| `src/services/llamaindex_rag.py` | Modal integration for RAG |
| `src/agents/code_executor_agent.py` | Modal sandbox execution |
| `src/utils/exceptions.py` | Modal-related exceptions |
| `src/tools/code_execution.py` | Modal code execution tool |
| `src/services/statistical_analyzer.py` | Modal statistical analysis |
| `src/mcp_tools.py` | Modal MCP tool wrappers |
| `src/agents/analysis_agent.py` | Modal analysis agent |

---

## Context

Modal was originally integrated for:
1. **Code Execution Sandbox**: Running untrusted code in isolated containers
2. **Statistical Analysis**: Offloading heavy statistical computations
3. **LlamaIndex RAG**: Premium embeddings with persistent storage

However, the project decided against Modal because:
- Added infrastructure complexity
- Free Tier doesn't need cloud functions
- Paid Tier uses OpenAI directly

---

## Recommended Fix

1. Remove Modal-related code from all affected files
2. Remove `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from config
3. Remove Modal from dependencies in `pyproject.toml`
4. Update any documentation referencing Modal

---

## Impact If Not Fixed

- Confusion for new contributors
- Dead code in production
- Unnecessary dependencies
- Config settings that do nothing

---

## Test Plan

1. Remove Modal code
2. Run `make check` to ensure no breakage
3. Verify Free Tier and Paid Tier still work
4. Search codebase for any remaining Modal references

---

## Related

- `P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md` - Similar tech debt for Anthropic
- ARCHITECTURE.md - Current architecture excludes Modal
