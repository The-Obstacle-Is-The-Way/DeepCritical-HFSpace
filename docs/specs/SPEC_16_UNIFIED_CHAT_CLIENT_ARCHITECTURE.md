# SPEC_16: Unified Chat Client Architecture

**Status**: Proposed
**Priority**: P1 (Architectural Simplification)
**Issue**: Updates [#105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105), [#109](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/109)
**Created**: 2025-12-01
**Last Verified**: 2025-12-01 (line counts and imports verified against codebase)

## Summary

Eliminate the Simple Mode / Advanced Mode parallel universe by implementing a pluggable `ChatClient` architecture. This moves the system away from a hardcoded `OpenAIChatClient` namespace to a neutral `BaseChatClient` protocol, allowing the multi-agent framework to work with ANY LLM provider through a unified codebase.

## Strategic Goals

1.  **Namespace Neutrality**: Decouple the core orchestrator from the `OpenAI` namespace. The system should speak `ChatClient`, not `OpenAIChatClient`.
2.  **Full-Stack Provider Chain**: Prioritize providers that offer both LLM and Embeddings (OpenAI, Gemini, HuggingFace+Local) to ensure a unified environment.
3.  **Fragmentation Reduction**: Remove "LLM-only" providers (Anthropic) that force complex hybrid dependency chains (e.g., Anthropic LLM + OpenAI Embeddings).

## Problem Statement

### Current Architecture: Two Parallel Universes

```
User Query
    │
    ├── Has API Key? ──Yes──→ Advanced Mode (488 lines)
    │                         └── Microsoft Agent Framework
    │                         └── OpenAIChatClient (hardcoded dependency)
    │
    └── No API Key? ──────────→ Simple Mode (778 lines)
                                └── While-loop orchestration
                                └── Pydantic AI + HuggingFace
```

**Problems:**
1. **Double Maintenance**: 1,266 lines across two orchestrator systems.
2. **Namespace Lock-in**: The Advanced Orchestrator is tightly coupled to `OpenAIChatClient` (25 references across 5 files).
3. **Fragmented Chains**: Using Anthropic requires a "Frankenstein" chain (Anthropic LLM + OpenAI Embeddings).
4. **Testing Burden**: Two test suites, two CI paths.

## Proposed Solution: ChatClientFactory

### Architecture After Implementation

```
User Query
    │
    └──→ Advanced Mode (unified)
         └── Microsoft Agent Framework
         └── ChatClientFactory (Namespace Neutral):
             ├── OpenAIChatClient (Paid Tier: Best Performance)
             ├── GeminiChatClient (Alternative Tier: LLM + Embeddings)
             └── HuggingFaceChatClient (Free Tier: LLM + Local Embeddings)
```

### New Files

```
src/
├── clients/
│   ├── __init__.py
│   ├── base.py              # Re-export BaseChatClient (The neutral protocol)
│   ├── factory.py           # ChatClientFactory
│   ├── huggingface.py       # HuggingFaceChatClient
│   └── gemini.py            # GeminiChatClient [Future]
```

### ChatClientFactory Implementation

```python
# src/clients/factory.py
from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatClient
from src.utils.config import settings

def get_chat_client(
    provider: str | None = None,
    api_key: str | None = None,
) -> BaseChatClient:
    """
    Factory for creating chat clients.

    Auto-detection priority:
    1. Explicit provider parameter
    2. OpenAI key (Best Function Calling)
    3. Gemini key (Best Context/Cost)
    4. HuggingFace (Free Fallback)

    Args:
        provider: Force specific provider ("openai", "gemini", "huggingface")
        api_key: Override API key for the provider

    Returns:
        Configured BaseChatClient instance (Neutral Namespace)
    """
    # OpenAI (Standard)
    if provider == "openai" or (provider is None and settings.has_openai_key):
        return OpenAIChatClient(
            model_id=settings.openai_model,
            api_key=api_key or settings.openai_api_key,
        )

    # Gemini (High Performance Alternative) - REQUIRES config.py update first
    if provider == "gemini" or (provider is None and settings.has_gemini_key):
        from src.clients.gemini import GeminiChatClient
        return GeminiChatClient(
            model_id="gemini-2.0-flash",
            api_key=api_key or settings.gemini_api_key,
        )

    # Free Fallback (HuggingFace)
    from src.clients.huggingface import HuggingFaceChatClient
    return HuggingFaceChatClient(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
    )
```

### Changes to Advanced Orchestrator

```python
# src/orchestrators/advanced.py

# BEFORE (hardcoded namespace):
from agent_framework.openai import OpenAIChatClient

class AdvancedOrchestrator:
    def __init__(self, ...):
        self._chat_client = OpenAIChatClient(...)

# AFTER (neutral namespace):
from src.clients.factory import get_chat_client

class AdvancedOrchestrator:
    def __init__(self, chat_client=None, provider=None, api_key=None, ...):
        # The orchestrator no longer knows about OpenAI
        self._chat_client = chat_client or get_chat_client(
            provider=provider,
            api_key=api_key,
        )
```

---

## Technical Requirements

### BaseChatClient Protocol (Verified)

The `agent_framework.BaseChatClient` requires implementing **2 abstract methods**:

```python
class HuggingFaceChatClient(BaseChatClient):
    """Adapter for HuggingFace Inference API."""

    async def _inner_get_response(
        self,
        messages: list[ChatMessage],
        **kwargs
    ) -> ChatResponse:
        """Synchronous response generation."""
        ...

    async def _inner_get_streaming_response(
        self,
        messages: list[ChatMessage],
        **kwargs
    ) -> AsyncIterator[ChatResponseUpdate]:
        """Streaming response generation."""
        ...
```

### Required Config Changes

**BEFORE implementation**, add to `src/utils/config.py`:

```python
# Settings class additions:
gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")

@property
def has_gemini_key(self) -> bool:
    """Check if Gemini API key is available."""
    return bool(self.gemini_api_key)
```

---

## Files to Modify (Complete List)

### Category 1: OpenAIChatClient References (25 total)

| File | Lines | Changes Required |
|------|-------|------------------|
| `src/orchestrators/advanced.py` | 31, 70, 95, 101, 122 | Replace with `get_chat_client()` |
| `src/agents/magentic_agents.py` | 4, 17, 29, 58, 70, 117, 129, 161, 173 | Change type hints to `BaseChatClient` |
| `src/agents/retrieval_agent.py` | 5, 53, 62 | Change type hints to `BaseChatClient` |
| `src/agents/code_executor_agent.py` | 7, 43, 52 | Change type hints to `BaseChatClient` |
| `src/utils/llm_factory.py` | 19, 22, 35, 38, 42 | Merge into `clients/factory.py` |

### Category 2: Anthropic References (46 total - Issue #110)

| File | Refs | Changes Required |
|------|------|------------------|
| `src/agent_factory/judges.py` | 10 | Remove Anthropic imports and fallback |
| `src/utils/config.py` | 10 | Remove `anthropic_api_key`, `anthropic_model`, `has_anthropic_key` |
| `src/utils/llm_factory.py` | 10 | Remove Anthropic model creation |
| `src/app.py` | 12 | Remove Anthropic key detection and UI |
| `src/orchestrators/simple.py` | 2 | Remove Anthropic mentions |
| `src/agents/hypothesis_agent.py` | 1 | Update comment |

### Category 3: Files to Delete (Phase 3)

| File | Lines | Reason |
|------|-------|--------|
| `src/orchestrators/simple.py` | 778 | Replaced by unified Advanced Mode |
| `src/tools/search_handler.py` | 219 | Manager agent handles orchestration |

**Total deletion: ~997 lines**
**Total addition: ~400 lines (new clients)**
**Net: ~600 fewer lines, single architecture**

---

## Migration Plan

### Phase 1: Neutralize Namespace & Add HuggingFace
- [ ] Add `gemini_api_key` and `has_gemini_key` to `src/utils/config.py`
- [ ] Create `src/clients/` package
- [ ] Implement `HuggingFaceChatClient` adapter (~150 lines)
- [ ] Implement `ChatClientFactory` (~50 lines)
- [ ] Refactor `AdvancedOrchestrator` to use `get_chat_client()`
- [ ] Update type hints in `magentic_agents.py`, `retrieval_agent.py`, `code_executor_agent.py`
- [ ] Merge `llm_factory.py` functionality into `clients/factory.py`

### Phase 2: Simplify Provider Chain (Issue #110)
- [ ] Remove Anthropic from `judges.py` (10 refs)
- [ ] Remove Anthropic from `config.py` (10 refs)
- [ ] Remove Anthropic from `llm_factory.py` (10 refs)
- [ ] Remove Anthropic from `app.py` (12 refs)
- [ ] Update user-facing strings mentioning Anthropic
- [ ] (Future) Implement `GeminiChatClient` (~200 lines)

### Phase 3: Deprecate Simple Mode (Issue #105)
- [ ] Update `src/orchestrators/factory.py` to use unified `AdvancedOrchestrator`
- [ ] Delete `src/orchestrators/simple.py` (778 lines)
- [ ] Delete `src/tools/search_handler.py` (219 lines)
- [ ] Update tests to only test Advanced Mode
- [ ] Archive deleted files to `docs/archive/` for reference

---

## Why This is "Elegant"

1.  **One System**: We stop maintaining two parallel universes.
2.  **Dependency Injection**: The specific LLM provider is injected, not hardcoded.
3.  **Full Stack Alignment**: We prioritize providers (OpenAI, Gemini) that own the whole vertical (LLM + Embeddings), reducing environment complexity.

---

## Verification Checklist (For Implementer)

Before starting implementation, verify:

- [x] `agent_framework.BaseChatClient` exists (verified: `agent_framework._clients.BaseChatClient`)
- [x] Abstract methods: `_inner_get_response`, `_inner_get_streaming_response`
- [x] `agent_framework.ChatResponse`, `ChatResponseUpdate`, `ChatMessage` importable
- [x] `settings.has_openai_key` exists (line 118)
- [ ] `settings.has_gemini_key` **MUST BE ADDED** (does not exist)
- [ ] `settings.gemini_api_key` **MUST BE ADDED** (does not exist)

---

## References

- Microsoft Agent Framework: `agent_framework.BaseChatClient`
- Gemini API: [Embeddings + LLM](https://ai.google.dev/gemini-api/docs/embeddings)
- HuggingFace Inference: `huggingface_hub.InferenceClient`
- Issue #105: Deprecate Simple Mode
- Issue #109: Simplify Provider Architecture
- Issue #110: Remove Anthropic Provider Support
