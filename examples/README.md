# Examples

Demo scripts for DeepCritical functionality.

## 1. Search Demo (Phase 2)

Demonstrates parallel search across PubMed and Web. **No API keys required.**

```bash
uv run python examples/search_demo/run_search.py "metformin cancer"
```

## 2. Agent Demo (Phase 4)

Demonstrates the full search-judge-synthesize loop.

**Option A: Mock Mode (No Keys)**
Test the logic/mechanics without an LLM.
```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin cancer" --mock
```

**Option B: Real Mode (Requires Keys)**
Uses the real LLM Judge to evaluate evidence.
Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`.
```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin cancer"
```