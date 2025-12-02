# DeepBoner Architecture

> **Last Updated**: 2025-12-01

---

## How It Works (Simple Version)

```text
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED ARCHITECTURE                      │
│                                                              │
│   User provides API key?                                     │
│                                                              │
│   NO (Free Tier)              YES (Paid Tier)               │
│   ──────────────              ───────────────               │
│   HuggingFace backend         OpenAI backend                │
│   Qwen 2.5 72B (free)         GPT-5 (paid)                  │
│                                                              │
│   SAME orchestration logic for both                          │
│   ONE codebase, different LLM backends                       │
└─────────────────────────────────────────────────────────────┘
```

**That's it.** No "modes." Just: do you have an API key or not?

---

## Current Status

**Free Tier is BLOCKED** by upstream bug #2562.

Once [PR #2566](https://github.com/microsoft/agent-framework/pull/2566) merges:
1. Update `agent-framework` dependency
2. Free tier works
3. Done

---

## Framework Stack

DeepBoner uses TWO frameworks that work TOGETHER:

| Framework | What It Does | Where Used |
|-----------|--------------|------------|
| **Microsoft Agent Framework** | Multi-agent orchestration | `src/orchestrators/advanced.py` |
| **Pydantic AI** | Structured outputs, validation | `src/agent_factory/judges.py`, `src/agents/*.py` |

**They are NOT mutually exclusive.** Microsoft AF handles the orchestration (Manager → Search → Judge → Report). Pydantic AI handles structured outputs within those agents.

---

## LLM Backend Selection

Auto-detected by `src/clients/factory.py`:

```python
def get_chat_client():
    if settings.has_openai_key:
        return OpenAIChatClient(...)  # Paid tier
    else:
        return HuggingFaceChatClient(...)  # Free tier
```

| Condition | Backend | Model |
|-----------|---------|-------|
| User provides OpenAI key | OpenAI | GPT-5 |
| No API key provided | HuggingFace | Qwen 2.5 72B (free) |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/orchestrators/advanced.py` | Multi-agent orchestration (Microsoft AF) |
| `src/clients/factory.py` | Auto-selects LLM backend |
| `src/clients/huggingface.py` | HuggingFace adapter for free tier |
| `src/agent_factory/judges.py` | Judge logic (Pydantic AI) |
| `src/agents/*.py` | Individual agents (Pydantic AI) |

---

## What Was Deleted

`simple.py` (778 lines) was a SEPARATE orchestrator that created a "parallel universe." It's gone. Now there's ONE orchestrator with different backends.

---

## Upstream Blocker

**Bug:** Microsoft Agent Framework produces `repr()` garbage for tool-call-only messages.

**Fix:** [PR #2566](https://github.com/microsoft/agent-framework/pull/2566) - waiting to merge.

**Tracking:** [Issue #2562](https://github.com/microsoft/agent-framework/issues/2562)

---

## References

- [Pydantic AI](https://ai.pydantic.dev/) - Structured outputs framework
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - Multi-agent orchestration
- [AG-UI Protocol](https://www.copilotkit.ai/blog/introducing-pydantic-ai-integration-with-ag-ui) - How they integrate
