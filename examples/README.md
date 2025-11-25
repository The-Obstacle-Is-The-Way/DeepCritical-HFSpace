# DeepCritical Examples

Demo scripts demonstrating each phase of the drug repurposing research agent.

## Quick Start

```bash
# Run without API keys (mock modes available)
uv run python examples/embeddings_demo/run_embeddings.py
uv run python examples/full_stack_demo/run_full.py --mock

# Run with API keys (set OPENAI_API_KEY or ANTHROPIC_API_KEY)
uv run python examples/full_stack_demo/run_full.py "metformin cancer"
```

---

## 1. Search Demo (Phase 2)

Demonstrates parallel search across PubMed and Web sources. **No API keys required.**

```bash
uv run python examples/search_demo/run_search.py "metformin cancer"
```

**What it shows:**
- PubMed E-utilities search
- DuckDuckGo web search
- Scatter-gather orchestration
- Evidence model with citations

---

## 2. Agent Demo (Phase 4)

Demonstrates the search-judge-synthesize loop.

**Mock Mode (No API Keys):**
```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin cancer" --mock
```

**Real Mode (Requires API Keys):**
```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin cancer"
```

**What it shows:**
- Iterative search refinement
- LLM-based evidence assessment
- Synthesis generation
- Event streaming for UI updates

---

## 3. Magentic Demo (Phase 5)

Demonstrates multi-agent coordination using Microsoft Agent Framework.

```bash
# Requires OPENAI_API_KEY (Magentic uses OpenAI)
uv run python examples/orchestrator_demo/run_magentic.py "metformin cancer"
```

**What it shows:**
- MagenticBuilder workflow
- SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent coordination
- Manager-based orchestration

---

## 4. Embeddings Demo (Phase 6)

Demonstrates semantic search and deduplication. **No API keys required.**

```bash
uv run python examples/embeddings_demo/run_embeddings.py
```

**What it shows:**
- Text embedding with sentence-transformers
- ChromaDB vector storage
- Semantic similarity search
- Duplicate detection by meaning (not just URL)
- Cosine similarity calculations

---

## 5. Hypothesis Demo (Phase 7)

Demonstrates mechanistic hypothesis generation.

```bash
# Requires OPENAI_API_KEY or ANTHROPIC_API_KEY
uv run python examples/hypothesis_demo/run_hypothesis.py "metformin Alzheimer's"
uv run python examples/hypothesis_demo/run_hypothesis.py "sildenafil heart failure"
```

**What it shows:**
- Drug -> Target -> Pathway -> Effect reasoning
- Knowledge gap identification
- Search query suggestions for targeted research
- Confidence scoring

---

## 6. Full Stack Demo (Phases 1-8)

**The complete pipeline** - demonstrates all phases working together.

**Mock Mode (No API Keys):**
```bash
uv run python examples/full_stack_demo/run_full.py --mock
```

**Real Mode:**
```bash
uv run python examples/full_stack_demo/run_full.py "metformin Alzheimer's"
uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" -i 3
```

**What it shows:**
1. **Search** - PubMed + Web evidence collection
2. **Embeddings** - Semantic deduplication
3. **Hypothesis** - Mechanistic reasoning
4. **Judge** - Evidence quality assessment
5. **Report** - Structured scientific report generation

Output includes a publication-quality research report with:
- Executive summary
- Methodology
- Hypotheses tested (with support/contradict counts)
- Mechanistic and clinical findings
- Drug candidates
- Limitations
- Formatted references

---

## API Keys

| Example | Required Keys |
|---------|--------------|
| search_demo | None (optional NCBI_API_KEY for higher rate limits) |
| orchestrator_demo --mock | None |
| orchestrator_demo | OPENAI_API_KEY or ANTHROPIC_API_KEY |
| run_magentic | OPENAI_API_KEY |
| embeddings_demo | None |
| hypothesis_demo | OPENAI_API_KEY or ANTHROPIC_API_KEY |
| full_stack_demo --mock | None |
| full_stack_demo | OPENAI_API_KEY or ANTHROPIC_API_KEY |

---

## Architecture Overview

```
User Query
    |
    v
[Phase 2: Search] --> PubMed + Web
    |
    v
[Phase 6: Embeddings] --> Semantic Deduplication
    |
    v
[Phase 7: Hypothesis] --> Drug -> Target -> Pathway -> Effect
    |
    v
[Phase 3: Judge] --> "Is evidence sufficient?"
    |
    +---> NO --> Refine queries, loop back to Search
    |
    +---> YES --> Continue to Report
    |
    v
[Phase 8: Report] --> Structured Scientific Report
    |
    v
Final Output with Citations
```
