# DeepBoner Examples

**NO MOCKS. NO FAKE DATA. REAL SCIENCE.**

These demos run the REAL drug repurposing research pipeline with actual API calls.

---

## Prerequisites

You MUST have API keys configured:

```bash
# Copy the example and add your keys
cp .env.example .env

# Required (pick one):
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (higher PubMed rate limits):
NCBI_API_KEY=your-key
```

---

## Examples

### 1. Search Demo (No LLM Required)

Demonstrates REAL parallel search across PubMed, ClinicalTrials.gov, and Europe PMC.

```bash
uv run python examples/search_demo/run_search.py "metformin cancer"
```

**What's REAL:**
- Actual NCBI E-utilities API calls (PubMed)
- Actual ClinicalTrials.gov API calls
- Actual Europe PMC API calls (includes preprints)
- Real papers, real trials, real preprints

---

### 2. Embeddings Demo (No LLM Required)

Demonstrates REAL semantic search and deduplication.

```bash
uv run python examples/embeddings_demo/run_embeddings.py
```

**What's REAL:**
- Actual sentence-transformers model (all-MiniLM-L6-v2)
- Actual ChromaDB vector storage
- Real cosine similarity computations
- Real semantic deduplication

---

### 3. Orchestrator Demo (LLM Required)

Demonstrates the REAL search-judge-synthesize loop.

```bash
uv run python examples/orchestrator_demo/run_agent.py "metformin cancer"
uv run python examples/orchestrator_demo/run_agent.py "aspirin alzheimer" --iterations 5
```

**What's REAL:**
- Real PubMed + ClinicalTrials + Europe PMC searches
- Real LLM judge evaluating evidence quality
- Real iterative refinement based on LLM decisions
- Real research synthesis

---

### 4. Magentic Demo (OpenAI Required)

Demonstrates REAL multi-agent coordination using Microsoft Agent Framework.

```bash
# Requires OPENAI_API_KEY specifically
uv run python examples/orchestrator_demo/run_magentic.py "metformin cancer"
```

**What's REAL:**
- Real MagenticBuilder orchestration
- Real SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent
- Real manager-based coordination

---

### 5. Hypothesis Demo (LLM Required)

Demonstrates REAL mechanistic hypothesis generation.

```bash
uv run python examples/hypothesis_demo/run_hypothesis.py "metformin Alzheimer's"
uv run python examples/hypothesis_demo/run_hypothesis.py "sildenafil heart failure"
```

**What's REAL:**
- Real PubMed + Web search first
- Real embedding-based deduplication
- Real LLM generating Drug -> Target -> Pathway -> Effect chains
- Real knowledge gap identification

---

### 6. Full-Stack Demo (LLM Required)

**THE COMPLETE PIPELINE** - All phases working together.

```bash
uv run python examples/full_stack_demo/run_full.py "metformin Alzheimer's"
uv run python examples/full_stack_demo/run_full.py "sildenafil heart failure" -i 3
```

**What's REAL:**
1. Real PubMed + ClinicalTrials + Europe PMC evidence collection
2. Real embedding-based semantic deduplication
3. Real LLM mechanistic hypothesis generation
4. Real LLM evidence quality assessment
5. Real LLM structured scientific report generation

Output: Publication-quality research report with validated citations.

---

## API Key Requirements

| Example | LLM Required | Keys |
|---------|--------------|------|
| search_demo | No | Optional: `NCBI_API_KEY` |
| embeddings_demo | No | None |
| orchestrator_demo | Yes | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| run_magentic | Yes | `OPENAI_API_KEY` (Magentic requires OpenAI) |
| hypothesis_demo | Yes | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| full_stack_demo | Yes | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |

---

## Architecture

```text
User Query
    |
    v
[REAL Search] --> PubMed + ClinicalTrials + Europe PMC APIs
    |
    v
[REAL Embeddings] --> Actual sentence-transformers
    |
    v
[REAL Hypothesis] --> Actual LLM reasoning
    |
    v
[REAL Judge] --> Actual LLM assessment
    |
    +---> Need more? --> Loop back to Search
    |
    +---> Sufficient --> Continue
    |
    v
[REAL Report] --> Actual LLM synthesis
    |
    v
Publication-Quality Research Report
```

---

## Why No Mocks?

> "Authenticity is the feature."

Mocks belong in `tests/unit/`, not in demos. When you run these examples, you see:
- Real papers from real databases
- Real AI reasoning about real evidence
- Real scientific hypotheses
- Real research reports

This is what DeepBoner actually does. No fake data. No canned responses.
