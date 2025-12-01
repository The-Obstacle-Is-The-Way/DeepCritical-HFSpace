# SPEC_16: Unified Chat Client Architecture

**Status**: Proposed
**Priority**: P1 (Architectural Simplification)
**Issue**: Updates [#105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105), [#109](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/109)
**Created**: 2025-12-01

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
    ├── Has API Key? ──Yes──→ Advanced Mode (400 lines)
    │                         └── Microsoft Agent Framework
    │                         └── OpenAIChatClient (hardcoded dependency)
    │
    └── No API Key? ──────────→ Simple Mode (761 lines)
                                └── While-loop orchestration
                                └── Pydantic AI + HuggingFace
```

**Problems:**
1. **Double Maintenance**: 1,161 lines across two systems.
2. **Namespace Lock-in**: The Advanced Orchestrator is tightly coupled to `OpenAIChatClient`.
3. **Fragmented Chains**: Using Anthropic requires a "frankstein" chain (Anthropic LLM + OpenAI Embeddings).
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

    # Gemini (High Performance Alternative)
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

## Migration Plan

### Phase 1: Neutralize Namespace & Add HuggingFace
- [ ] Create `src/clients/` package.
- [ ] Implement `HuggingFaceChatClient` adapter.
- [ ] Implement `ChatClientFactory`.
- [ ] Refactor `AdvancedOrchestrator` to use `get_chat_client()`.
- [ ] Update strict typing to use `BaseChatClient` instead of `OpenAIChatClient`.

### Phase 2: Simplify Provider Chain
- [ ] Remove `Anthropic` references (Issue #110).
- [ ] (Future) Implement `GeminiChatClient` to support Google's full stack.

### Phase 3: Deprecate Simple Mode
- [ ] Update `src/orchestrators/factory.py` to use unified `AdvancedOrchestrator`.
- [ ] Delete `src/orchestrators/simple.py`.
- [ ] Delete `src/tools/search_handler.py`.

## Why This is "Elegant"

1.  **One System**: We stop maintaining two parallel universes.
2.  **Dependency Injection**: The specific LLM provider is injected, not hardcoded.
3.  **Full Stack Alignment**: We prioritize providers (OpenAI, Gemini) that own the whole vertical (LLM + Embeddings), reducing environment complexity.

## References

- Microsoft Agent Framework: `agent_framework.BaseChatClient`
- Gemini API: [Embeddings + LLM](https://ai.google.dev/gemini-api/docs/embeddings)
- HuggingFace Inference: `huggingface_hub.InferenceClient`