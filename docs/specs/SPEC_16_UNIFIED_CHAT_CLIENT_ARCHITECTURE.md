# SPEC_16: Unified Architecture

**Status**: BLOCKED - Waiting for upstream PR #2566
**Priority**: P0
**Issue**: [#105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105)
**Created**: 2025-12-01

---

## The Architecture (No Bullshit Version)

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

**No "modes."** Just: do you have an API key or not?

---

## Framework Stack

DeepBoner uses TWO frameworks that work TOGETHER:

| Framework | Role | Files |
|-----------|------|-------|
| **Microsoft Agent Framework** | Multi-agent ORCHESTRATION | `src/orchestrators/advanced.py` |
| **Pydantic AI** | Structured OUTPUTS & validation | `src/agent_factory/judges.py`, `src/agents/*.py` |

### Why Both?

- **Microsoft AF** handles: Manager → Search → Judge → Report agent coordination
- **Pydantic AI** handles: Structured responses, type validation, schema enforcement

They are **NOT mutually exclusive**. They are **complementary**:
- Microsoft AF = the highway system (routes agents)
- Pydantic AI = the cargo containers (structures data)

### Current Integration

| Component | Framework | Purpose |
|-----------|-----------|---------|
| `AdvancedOrchestrator` | Microsoft AF | Coordinates multi-agent workflow |
| `JudgeAssessment` | Pydantic AI | Structured judge output with validation |
| `Evidence`, `Citation` | Pydantic | Validated data models |
| Agent tool calling | Microsoft AF | Function execution |
| Agent structured output | Pydantic AI | Response validation |

---

## LLM Backend Selection

Auto-detected by `src/clients/factory.py`:

| Condition | Backend | Model |
|-----------|---------|-------|
| User provides OpenAI key | OpenAI | GPT-5 |
| No API key | HuggingFace | Qwen 2.5 72B (free) |

---

## Current Blocker

**Upstream Bug #2562**: Microsoft Agent Framework produces `repr()` garbage for tool-call-only messages.

**Fix**: [PR #2566](https://github.com/microsoft/agent-framework/pull/2566) - waiting for merge.

**Once merged**:
1. `uv add agent-framework@latest`
2. Verify free tier works
3. Done

---

## What Was Deleted

`simple.py` (778 lines) was a SEPARATE orchestrator that created a parallel universe:
- Used Pydantic AI directly for LLM calls
- Had its own while-loop orchestration
- Duplicated search/judge logic

Now there's ONE orchestrator with different backends.

---

## Files

| File | Framework | Purpose |
|------|-----------|---------|
| `src/orchestrators/advanced.py` | Microsoft AF | Multi-agent orchestration |
| `src/clients/factory.py` | - | Auto-selects LLM backend |
| `src/clients/huggingface.py` | - | HuggingFace adapter (free tier) |
| `src/agent_factory/judges.py` | Pydantic AI | Structured judge assessments |
| `src/agents/report_agent.py` | Pydantic AI | Structured report generation |
| `src/utils/models.py` | Pydantic | Data models (Evidence, Citation) |

---

## References

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - Multi-agent orchestration
- [Pydantic AI](https://ai.pydantic.dev/) - Structured outputs framework
- [Multi-Agent Research System with Pydantic](https://www.analyticsvidhya.com/blog/2025/03/multi-agent-research-assistant-system-using-pydantic/) - Architecture pattern
- [AG-UI Protocol](https://www.copilotkit.ai/blog/introducing-pydantic-ai-integration-with-ag-ui) - How frameworks integrate
