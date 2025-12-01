# SPEC_16: Unified Chat Client Architecture

**Status**: Proposed
**Priority**: P0 (Fixes Critical Bug #113)
**Issue**: Updates [#105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105), [#109](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/109), **[#113](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/113)** (P0 Bug)
**Created**: 2025-12-01
**Last Updated**: 2025-12-01

---

## ⚠️ CRITICAL CLARIFICATION: Integration, Not Deletion

**This spec INTEGRATES Simple Mode's free-tier capability into Advanced Mode.**

| What We're Doing | What We're NOT Doing |
|------------------|----------------------|
| ✅ Integrating HuggingFace support into Advanced Mode | ❌ Removing free-tier capability |
| ✅ Unifying two parallel implementations into one | ❌ Breaking functionality for users without API keys |
| ✅ Deleting redundant orchestration CODE | ❌ Deleting the CAPABILITY that code provided |
| ✅ Making Advanced Mode work with ANY provider | ❌ Locking users into paid-only tiers |

**After this spec:**
- Users WITH OpenAI key → Advanced Mode (OpenAI backend) ✅
- Users WITHOUT any key → Advanced Mode (HuggingFace backend) ✅ **SAME CAPABILITY, UNIFIED ARCHITECTURE**

---

## Summary

Unify Simple Mode and Advanced Mode into a **single orchestration system** by:

1. **Renaming the namespace**: `OpenAIChatClient` → `BaseChatClient` (neutral protocol)
2. **Creating an adapter**: `HuggingFaceChatClient` implements `BaseChatClient`
3. **Retiring parallel code**: Simple Mode's while-loop becomes unnecessary

The result: **One codebase, multiple providers, zero parallel universes.**

> **🔥 P0 Bug Fix**: This also resolves Issue #113. Simple Mode's `_should_synthesize()` has a bug that ignores forced synthesis signals. Advanced Mode's Manager agent handles termination correctly. By integrating, the bug disappears.

---

## The Integration Concept

### Before: Two Parallel Universes (Current)

```text
User Query
    │
    ├── Has API Key? ──Yes──→ Advanced Mode (488 lines)
    │                         └── Microsoft Agent Framework
    │                         └── OpenAIChatClient (hardcoded) ◄── THE BOTTLENECK
    │
    └── No API Key? ──────────→ Simple Mode (778 lines)
                                └── While-loop orchestration (SEPARATE CODE)
                                └── Pydantic AI + HuggingFace
```

**Problem**: Same capability, two implementations, double maintenance, P0 bug in Simple Mode.

### After: Unified Architecture (This Spec)

```text
User Query
    │
    └──→ Advanced Mode (unified) ◄── ONE SYSTEM FOR ALL USERS
         └── Microsoft Agent Framework
         └── get_chat_client() returns: ◄── NAMESPACE NEUTRAL
             │
             ├── OpenAIChatClient      (if OpenAI key present)
             ├── GeminiChatClient      (if Gemini key present) [Future]
             └── HuggingFaceChatClient (fallback - FREE TIER) ◄── INTEGRATED!
```

**Result**: Free-tier users get the SAME Advanced Mode experience, just with HuggingFace as the LLM backend.

---

## What Gets Integrated vs Retired

### ✅ INTEGRATED (Capability Preserved)

| Simple Mode Component | Integration Target | How |
|-----------------------|-------------------|-----|
| HuggingFace LLM calls | `HuggingFaceChatClient` | New adapter (~150 lines) |
| Free-tier access | `get_chat_client()` factory | Auto-selects HF when no key |
| Search tools (PubMed, etc.) | Already shared | `src/agents/tools.py` |
| Evidence models | Already shared | `src/utils/models.py` |

### 🗑️ RETIRED (Redundant Code Removed)

| Simple Mode Component | Why Retired | Replacement in Advanced Mode |
|-----------------------|-------------|------------------------------|
| While-loop orchestration | Redundant | Manager agent orchestrates |
| `_should_synthesize()` thresholds | **BUGGY** (P0 #113) | Manager agent signals |
| `SearchHandler` scatter-gather | Redundant | SearchAgent handles this |
| `JudgeHandler` | Redundant | JudgeAgent handles this |

**Key insight**: We're not losing functionality. We're consolidating two implementations of the SAME functionality into one.

---

## Technical Implementation

### The Single Change That Enables Unification

```python
# BEFORE (hardcoded to OpenAI):
from agent_framework.openai import OpenAIChatClient

class AdvancedOrchestrator:
    def __init__(self, ...):
        self._chat_client = OpenAIChatClient(...)  # ❌ Only OpenAI works

# AFTER (neutral - any provider):
from agent_framework import BaseChatClient
from src.clients.factory import get_chat_client

class AdvancedOrchestrator:
    def __init__(self, ...):
        self._chat_client = get_chat_client()  # ✅ OpenAI, Gemini, OR HuggingFace
```

### HuggingFaceChatClient Adapter

```python
# src/clients/huggingface.py
from agent_framework import BaseChatClient, ChatMessage, ChatResponse
from huggingface_hub import InferenceClient

class HuggingFaceChatClient(BaseChatClient):
    """Adapter that makes HuggingFace work with Microsoft Agent Framework."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-70B-Instruct"):
        self._client = InferenceClient(model=model_id)
        self._model_id = model_id

    async def _inner_get_response(
        self,
        messages: list[ChatMessage],
        **kwargs
    ) -> ChatResponse:
        """Convert HuggingFace response to Agent Framework format."""
        # Convert messages to HF format
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Call HuggingFace
        response = self._client.chat_completion(messages=hf_messages)

        # Convert back to Agent Framework format
        return ChatResponse(
            content=response.choices[0].message.content,
            # ... other fields
        )

    async def _inner_get_streaming_response(self, ...):
        """Streaming version."""
        ...
```

### ChatClientFactory

```python
# src/clients/factory.py
from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatClient
from src.utils.config import settings

def get_chat_client(provider: str | None = None) -> BaseChatClient:
    """
    Factory that returns the appropriate chat client.

    Priority:
    1. OpenAI (if key available) - Best function calling, GPT-5
    2. Gemini (if key available) - Good alternative [Future]
    3. HuggingFace (always available) - FREE TIER FALLBACK
    """
    if provider == "openai" or (provider is None and settings.has_openai_key):
        return OpenAIChatClient(
            model_id=settings.openai_model,  # gpt-5
            api_key=settings.openai_api_key,
        )

    # Future: Gemini support
    # if settings.has_gemini_key:
    #     return GeminiChatClient(...)

    # FREE TIER: HuggingFace (no API key required for public models)
    from src.clients.huggingface import HuggingFaceChatClient
    return HuggingFaceChatClient(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
    )
```

---

## Why This Fixes P0 Bug #113

### The Bug (Simple Mode)

```python
# src/orchestrators/simple.py - THE BUG
def _should_synthesize(self, assessment, ...):
    # When HF fails, judge returns: score=0, confidence=0.1, recommendation="synthesize"

    if assessment.sufficient and assessment.recommendation == "synthesize":
        if combined_score >= 10:  # ❌ 0 >= 10 is FALSE
            return True

    if confidence >= 0.5:  # ❌ 0.1 >= 0.5 is FALSE
        return True, "emergency"

    return False, "continue_searching"  # ❌ LOOPS FOREVER
```

### The Fix (Advanced Mode - Already Works Correctly)

```python
# Advanced Mode doesn't have this bug because:
# 1. JudgeAgent says "SUFFICIENT EVIDENCE" in natural language
# 2. Manager agent understands this and delegates to ReportAgent
# 3. No hardcoded thresholds to bypass

# The Manager agent prompt (src/orchestrators/advanced.py:152):
"""
When JudgeAgent says "SUFFICIENT EVIDENCE" or "STOP SEARCHING":
→ IMMEDIATELY delegate to ReportAgent for synthesis
"""
```

**By integrating Simple Mode's capability into Advanced Mode, the bug disappears** because Advanced Mode's termination logic works correctly.

---

## Migration Plan

### Phase 1: Create HuggingFaceChatClient (Enables Integration)

- [ ] Create `src/clients/` package
- [ ] Implement `HuggingFaceChatClient` (~150 lines)
  - Extends `agent_framework.BaseChatClient`
  - Wraps `huggingface_hub.InferenceClient.chat_completion()`
  - Implements required abstract methods
- [ ] Implement `get_chat_client()` factory (~50 lines)
- [ ] Add unit tests

**Exit Criteria**: `get_chat_client()` returns working HuggingFace client when no API key.

### Phase 2: Integrate into Advanced Mode (Fixes P0 Bug)

- [ ] Update `AdvancedOrchestrator` to use `get_chat_client()`
- [ ] Update `magentic_agents.py` type hints: `OpenAIChatClient` → `BaseChatClient`
- [ ] Update `orchestrators/factory.py` to always return `AdvancedOrchestrator`
- [ ] Update `app.py` to remove mode toggle (everyone gets Advanced Mode)
- [ ] Archive `simple.py` to `docs/archive/` (for reference)
- [ ] Migrate Simple Mode tests to Advanced Mode tests

**Exit Criteria**: Free-tier users get Advanced Mode with HuggingFace backend. P0 bug gone.

### Phase 3: Cleanup (Optional)

- [ ] Remove Anthropic provider code (Issue #110)
- [ ] Add Gemini support (Issue #109)
- [ ] Delete archived files after verification period

---

## Files Changed

### New Files (~200 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `src/clients/__init__.py` | ~10 | Package exports |
| `src/clients/factory.py` | ~50 | `get_chat_client()` |
| `src/clients/huggingface.py` | ~150 | HuggingFace adapter |

### Modified Files

| File | Change |
|------|--------|
| `src/orchestrators/advanced.py` | Use `get_chat_client()` instead of `OpenAIChatClient` |
| `src/orchestrators/factory.py` | Always return `AdvancedOrchestrator` |
| `src/agents/magentic_agents.py` | Type hints: `OpenAIChatClient` → `BaseChatClient` |
| `src/app.py` | Remove mode toggle, always use Advanced |

### Archived Files (NOT deleted from git history)

| File | Lines | Reason |
|------|-------|--------|
| `src/orchestrators/simple.py` | 778 | Functionality INTEGRATED, code retired |
| `src/tools/search_handler.py` | 219 | Manager agent handles this now |

---

## Verification Checklist

### Technical Prerequisites (Verified ✅)

- [x] `agent_framework.BaseChatClient` exists
- [x] Abstract methods: `_inner_get_response`, `_inner_get_streaming_response`
- [x] `huggingface_hub.InferenceClient.chat_completion()` exists
- [x] `chat_completion()` has `tools` parameter (verified in 0.36.0)
- [x] HuggingFace supports Llama 3.1 70B via free inference
- [x] **Dependency pinned**: `huggingface-hub>=0.24.0` in pyproject.toml (required for stable tool calling)

### Capability Preservation Checklist

After implementation, verify:

- [ ] User with OpenAI key → Gets Advanced Mode with OpenAI (GPT-5)
- [ ] User with NO key → Gets Advanced Mode with HuggingFace (Llama 3.1 70B)
- [ ] Free-tier search works (PubMed, ClinicalTrials, EuropePMC)
- [ ] Free-tier synthesis works (LLM generates report)
- [ ] No more "continue_searching" infinite loops (P0 bug fixed)

---

## Implementation Notes (From Independent Audit)

### Dependency Requirement ✅ FIXED

The `huggingface-hub` package must be `>=0.24.0` for stable `chat_completion` with tools support.

```toml
# pyproject.toml - ALREADY UPDATED
"huggingface-hub>=0.24.0",  # Required for stable chat_completion with tools
```

### Llama 3.1 Prompt Considerations ⚠️

The Manager agent prompt in `AdvancedOrchestrator._create_task_prompt()` was optimized for GPT-5. When using Llama 3.1 70B via HuggingFace, the prompt **may need tuning** to ensure strict adherence to delegation logic.

**Potential issue**: Llama 3.1 may not immediately delegate to ReportAgent when JudgeAgent says "SUFFICIENT EVIDENCE".

**Mitigation**: During implementation, test with HuggingFace backend and add reinforcement phrases if needed:
- "You MUST delegate to ReportAgent when you see SUFFICIENT EVIDENCE"
- "Do NOT continue searching after Judge approves"

This is a **runtime verification** task, not a spec change.

---

## References

- Microsoft Agent Framework: `agent_framework.BaseChatClient`
- HuggingFace Inference: `huggingface_hub.InferenceClient`
- Issue #105: Deprecate Simple Mode → **Reframe as "Integrate Simple Mode"**
- Issue #109: Simplify Provider Architecture
- Issue #110: Remove Anthropic Provider Support
- Issue #113: P0 Bug - Simple Mode ignores forced synthesis
