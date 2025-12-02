# DeepBoner Architecture

> **Last Updated**: 2025-12-01
> **Status**: Unified Architecture IN PROGRESS (blocked by upstream #2562)

---

## Current State

### Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  Orchestrator Factory               │
│            src/orchestrators/factory.py             │
│                                                     │
│  create_orchestrator() → ALWAYS returns Advanced    │
│  _determine_mode() → "simple" deprecated → advanced │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Advanced Orchestrator                  │
│            src/orchestrators/advanced.py            │
│                                                     │
│  Microsoft Agent Framework (MagenticBuilder)        │
│  Multi-agent: Manager, Search, Judge, Report        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Chat Client Factory                    │
│              src/clients/factory.py                 │
│                                                     │
│  get_chat_client() auto-selects:                    │
│  ├── OpenAI (if key present) → OpenAIChatClient     │
│  └── HuggingFace (free fallback) → HuggingFaceChatClient
└─────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/orchestrators/factory.py` | Creates orchestrators | ✅ Unified |
| `src/orchestrators/advanced.py` | Microsoft Agent Framework orchestration | ✅ Working (OpenAI) |
| `src/clients/factory.py` | Auto-selects chat client | ✅ Unified |
| `src/clients/huggingface.py` | HuggingFace adapter for Agent Framework | ✅ Created |
| `src/orchestrators/simple.py` | **DELETED** | ❌ Gone (premature) |

### Current Problem

**Upstream Bug #2562**: Microsoft Agent Framework produces `repr()` strings instead of message text for tool-call-only messages.

```python
# In Microsoft Agent Framework (_invoke_agent)
text = last.text or str(last)  # BUG: str(last) = "<ChatMessage object at 0x...>"
```

**Result**: Free Tier (Advanced + HuggingFace) shows garbage output:
```
📚 **SEARCH_COMPLETE**: searcher: <agent_framework._types.ChatMessage object at 0x7fd3f8617b10>
```

**Upstream Fix**: PR #2566 submitted, waiting for merge.

---

## The Goal: Unified Architecture

### Vision

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIFIED ORCHESTRATOR                       │
│                                                              │
│  ONE codebase handles ALL tiers                              │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   Free Tier     │    │   Paid Tier     │                 │
│  │   (no API key)  │    │   (OpenAI key)  │                 │
│  └────────┬────────┘    └────────┬────────┘                 │
│           │                      │                           │
│           ▼                      ▼                           │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │  HuggingFace    │    │    OpenAI       │                 │
│  │  ChatClient     │    │   ChatClient    │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                              │
│  SAME orchestration logic, DIFFERENT LLM backends            │
└─────────────────────────────────────────────────────────────┘
```

### NOT Two Parallel Universes

**WRONG** (what we had before):
```
├── Simple Mode (778 lines) - while-loop orchestration
│   └── HuggingFace (free)
│
└── Advanced Mode (488 lines) - Agent Framework
    └── OpenAI (paid only)
```

**CORRECT** (unified architecture):
```
└── Advanced Mode (UNIFIED)
    ├── HuggingFace backend (free tier)
    └── OpenAI backend (paid tier)
```

### What "Simple Mode INTEGRATED" Means

| Aspect | Old Simple Mode | Integrated in Advanced Mode |
|--------|-----------------|----------------------------|
| **Free tier access** | Via separate orchestrator | Via HuggingFaceChatClient |
| **Search tools** | SearchHandler | SearchAgent |
| **Judge logic** | JudgeHandler | JudgeAgent |
| **Termination** | `_should_synthesize()` thresholds | Manager agent signals |
| **Synthesis** | Inline in orchestrator | ReportAgent |

The CAPABILITY is preserved. The REDUNDANT CODE is gone.

---

## Path Forward

### Scenario A: Upstream PR #2566 Merges (Expected)

1. **Update `agent-framework` dependency** to version with fix
2. **Verify** Advanced + HuggingFace produces clean output
3. **Done** - Unified architecture complete

```bash
# After upstream merges:
uv add agent-framework@latest  # or specific version with fix
uv run pytest tests/  # Verify
```

### Scenario B: Upstream PR #2566 Delayed (Fallback)

If upstream takes too long, we can apply the fix locally:

1. **Fork agent-framework** or vendor the fix
2. **Apply the one-line fix**:
   ```python
   # In agent_framework/_agent.py (_invoke_agent method)
   # BEFORE:
   text = last.text or str(last)
   # AFTER:
   text = last.text or ""
   ```
3. **Test locally** with patched framework
4. **Switch back to upstream** once merged

### Scenario C: Complete Re-Implementation (Not Recommended)

If upstream is abandoned or unresponsive:

1. Implement our own agent orchestration
2. Remove Microsoft Agent Framework dependency
3. Use `HuggingFaceChatClient` directly with custom orchestration

**NOT RECOMMENDED** because:
- Significant effort
- Lose Microsoft's framework benefits
- They're already fixing it (PR #2566)

---

## File Structure

```
src/
├── orchestrators/
│   ├── factory.py        # create_orchestrator() → UNIFIED
│   ├── advanced.py       # AdvancedOrchestrator (main)
│   ├── hierarchical.py   # HierarchicalOrchestrator (experimental)
│   ├── langgraph_orchestrator.py  # LangGraph (experimental)
│   └── base.py           # Protocols
│
├── clients/
│   ├── factory.py        # get_chat_client() → auto-selects
│   ├── huggingface.py    # HuggingFaceChatClient
│   └── base.py           # Protocols
│
├── agents/
│   ├── tools.py          # PubMed, ClinicalTrials, EuropePMC
│   └── magentic_agents.py  # Agent definitions
│
└── agent_factory/
    └── judges.py         # JudgeHandler (for reference)
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [SPEC_16](specs/SPEC_16_UNIFIED_CHAT_CLIENT_ARCHITECTURE.md) | Unified architecture spec |
| [P1 Simple Mode Bug](bugs/P1_SIMPLE_MODE_REMOVED_BREAKS_FREE_TIER_UX.md) | Why free tier is broken |
| [Issue #105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105) | GitHub tracking |
| [Issue #113](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/113) | Related bug |
| [Upstream #2562](https://github.com/microsoft/agent-framework/issues/2562) | Framework bug |
| [Upstream PR #2566](https://github.com/microsoft/agent-framework/pull/2566) | Framework fix |

---

## Summary

| Question | Answer |
|----------|--------|
| **Current state?** | Advanced Mode only, Simple Mode deleted |
| **Free tier works?** | No - blocked by upstream repr bug |
| **The goal?** | ONE unified architecture, not parallel universes |
| **Simple Mode deleted?** | Yes, but CAPABILITY is integrated via HuggingFaceChatClient |
| **What's blocking?** | Upstream PR #2566 needs to merge |
| **When fixed?** | Update agent-framework, verify, done |
