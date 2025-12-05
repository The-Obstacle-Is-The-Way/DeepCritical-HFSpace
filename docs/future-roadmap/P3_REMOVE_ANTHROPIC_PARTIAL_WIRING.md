# P3 Tech Debt: Remove Anthropic Partial Wiring

**Date**: 2025-12-03
**Status**: DONE
**Severity**: P3 (Tech Debt / Simplification)
**Component**: Architecture / Provider Integration

---

## Summary

Remove all Anthropic-related code, configuration, and references from the codebase. Anthropic is partially wired but **not fully threaded through the architecture**, creating confusion and half-implemented code paths.

---

## Rationale

### 1. Anthropic Does NOT Provide Embeddings

Our architecture requires embeddings for:
- RAG (LlamaIndex/ChromaDB)
- Evidence deduplication
- Semantic search

Anthropic only provides chat completion, not embeddings. This means even with a working Anthropic chat client, users would need a **second provider** for embeddings, breaking the unified experience.

### 2. Partial Implementation Creates Confusion

Current state:
- `settings.anthropic_api_key` exists ✅
- `settings.has_anthropic_key` property exists ✅
- `settings.anthropic_model` configured ✅
- `AnthropicChatClient` for agent_framework **DOES NOT EXIST** ❌
- Code raises `NotImplementedError` when Anthropic detected ❌

This half-state causes:
- User confusion ("Why doesn't my Anthropic key work?")
- Developer confusion ("Is Anthropic supported or not?")
- Dead code paths that need maintenance

### 3. Unified Architecture Principle

**Principle**: Only support providers that work **end-to-end** through the entire stack:

```
Provider Requirements:
├── Chat Completion (for agents)     ✅ Required
├── Function/Tool Calling            ✅ Required
├── Embeddings (for RAG)             ✅ Required
└── Streaming                        ✅ Required
```

| Provider | Chat | Tools | Embeddings | Streaming | Status |
|----------|------|-------|------------|-----------|--------|
| OpenAI | ✅ | ✅ | ✅ | ✅ | **KEEP** |
| HuggingFace | ✅ | ✅ | ✅ (local) | ✅ | **KEEP** |
| Gemini | ✅ | ✅ | ✅ | ✅ | Future (Phase 4) |
| Anthropic | ✅ | ✅ | ❌ | ✅ | **REMOVE** |

---

## Files to Clean Up

### Configuration
- [ ] `src/utils/config.py` - Remove `anthropic_api_key`, `anthropic_model`, `has_anthropic_key`

### Client Factory
- [ ] `src/clients/factory.py` - Remove Anthropic detection and `NotImplementedError`

### Legacy Code (pydantic-ai based)
- [ ] `src/utils/llm_factory.py` - Remove `AnthropicModel`, `AnthropicProvider` imports and handling
- [ ] `src/agent_factory/judges.py` - Remove Anthropic model selection

### App/UI
- [ ] `src/app.py` - Remove `has_anthropic_key` checks and "Anthropic from env" backend info

### Documentation
- [ ] `CLAUDE.md` - Update LLM provider list
- [ ] `AGENTS.md` - Update LLM provider list
- [ ] `GEMINI.md` - Update LLM provider list

### Tests
- [ ] `tests/unit/clients/test_chat_client_factory.py` - Remove Anthropic test cases
- [ ] `tests/unit/utils/test_config.py` - Remove Anthropic config tests

---

## Code Snippets to Remove

### `src/utils/config.py`
```python
# REMOVE these lines:
anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
anthropic_model: str = Field(
    default="claude-sonnet-4-5-20250929", description="Anthropic model"
)

@property
def has_anthropic_key(self) -> bool:
    """Check if Anthropic API key is available."""
    return bool(self.anthropic_api_key)
```

### `src/clients/factory.py`
```python
# REMOVE these lines:
if api_key.startswith("sk-ant-"):
    normalized = "anthropic"

if normalized == "anthropic":
    raise NotImplementedError(
        "Anthropic client not yet implemented. "
        "Use OpenAI key (sk-...) or leave empty for free HuggingFace tier."
    )
```

### `src/app.py`
```python
# REMOVE these lines:
elif settings.has_anthropic_key:
    backend_info = "Paid API (Anthropic from env)"

has_anthropic = settings.has_anthropic_key
has_paid_key = has_openai or has_anthropic or bool(user_api_key)
# Change to:
has_paid_key = has_openai or bool(user_api_key)
```

---

## Migration Notes

### For Users with Anthropic Keys

If users have `ANTHROPIC_API_KEY` set in their environment:
1. It will be **silently ignored** (not an error)
2. System falls through to HuggingFace free tier
3. Users should use `OPENAI_API_KEY` instead for paid tier

### Future Consideration

If Anthropic adds embeddings API in the future, we can re-add support. But until then, partial support creates more confusion than value.

---

## Definition of Done

- [ ] All Anthropic references removed from `src/`
- [ ] All Anthropic tests removed or updated
- [ ] Documentation updated to reflect supported providers: OpenAI, HuggingFace, (future: Gemini)
- [ ] `make check` passes (lint, typecheck, tests)
- [ ] PR reviewed and merged

---

## Related Documents

- `P2_7B_MODEL_GARBAGE_OUTPUT.md` - Current free tier model quality issues
- `HF_FREE_TIER_ANALYSIS.md` - HuggingFace provider routing analysis
- `CLAUDE.md` - Agent context with provider documentation
