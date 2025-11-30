# Embeddings Brainstorm - Conclusions

**Date**: November 2025
**Status**: CLOSED - Conclusions reached, no action needed

---

## The Question

Should DeepBoner implement:
1. Internal codebase embeddings/ingestion pipeline?
2. mGREP for internal tool selection?
3. Self-knowledge components for agents?

## The Answer: NO

After research and first-principles analysis, the conclusion is clear:

### Why Not Internal Embeddings/Ingestion

```text
DeepBoner's Core Task:
┌─────────────────────────────────────────────────────────┐
│  User Query: "Evidence for testosterone in HSDD?"       │
│                         ↓                               │
│  1. Search PubMed, ClinicalTrials, Europe PMC          │
│  2. Judge: Is evidence sufficient?                      │
│  3. Synthesize: Generate report                         │
│                         ↓                               │
│  Output: Research report with citations                 │
└─────────────────────────────────────────────────────────┘

Does ANY step require self-knowledge of codebase? NO.
```

### Why Not mGREP for Tool Selection

| Approach | Complexity | Accuracy |
|----------|------------|----------|
| Embeddings + mGREP for tool selection | High | Medium (semantic similarity ≠ correct tool) |
| Direct prompting with tool descriptions | Low | High (LLM reasons about applicability) |

**No real agent system uses embeddings for tool selection.** All major frameworks (LangChain, OpenAI, Anthropic, Magentic) use prompt-based tool selection because:
1. LLMs are already doing semantic matching internally
2. Tool count is small (5-20) - fits easily in context
3. Prompts allow reasoning, not just similarity

### What We Already Have

DeepBoner already uses embeddings for the **right thing**: research evidence retrieval.
- `src/services/embeddings.py` - ChromaDB + sentence-transformers
- `src/services/llamaindex_rag.py` - OpenAI embeddings for premium tier

### The Real Priority

Instead of internal embeddings/mGREP, focus on:
1. **Deduplication** across PubMed/Europe PMC/OpenAlex
2. **Outcome measures** from ClinicalTrials.gov
3. **Citation graph traversal** via OpenAlex

See: `TOOL_ANALYSIS_CRITICAL.md` for detailed improvement roadmap.

---

## Research Sources

- [SICA Paper (ICLR 2025)](https://arxiv.org/abs/2504.15228) - Self-improving agents
- [Gödel Agent (ACL 2025)](https://arxiv.org/abs/2410.04444) - Recursive self-modification
- [Introspection Paradox (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.352/) - Self-knowledge can hurt performance
- [Anthropic Introspection Research](https://www.anthropic.com/research/introspection) - ~20% accuracy on genuine introspection

---

*This document is closed. The conclusion is: don't implement internal embeddings/mGREP for this use case.*
