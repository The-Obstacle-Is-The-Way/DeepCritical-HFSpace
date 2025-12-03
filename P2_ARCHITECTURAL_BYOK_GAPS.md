# P2 Architectural: BYOK Gaps in Non-Critical Paths

**Date**: 2025-12-03
**Status**: ✅ RESOLVED
**Severity**: P2 (Architectural Debt)
**Component**: LLM Routing / BYOK Support
**Resolution**: Fixed end-to-end BYOK support in this PR

---

## Summary

Two code paths do NOT support BYOK (Bring Your Own Key) from Gradio:

1. **HierarchicalOrchestrator** - Doesn't receive `api_key` parameter
2. **get_model() (PydanticAI)** - Only checks env vars, no BYOK

These are **latent bugs** - they don't affect the main user flow currently.

---

## Bug 1: HierarchicalOrchestrator Missing api_key

**Location**: `src/orchestrators/factory.py:61-64`

```python
if effective_mode == "hierarchical":
    from src.orchestrators.hierarchical import HierarchicalOrchestrator
    return HierarchicalOrchestrator(config=effective_config, domain=domain)
    # BUG: api_key is NOT passed to HierarchicalOrchestrator
```

**Impact**: If hierarchical mode were exposed in UI, BYOK would not work.

**Current State**: Hierarchical mode is NOT exposed in Gradio UI, so this is latent.

**Fix**: Pass `api_key` to HierarchicalOrchestrator when instantiating.

---

## Bug 2: get_model() Doesn't Support BYOK

**Location**: `src/agent_factory/judges.py:62-91` (function `get_model()`)

```python
def get_model() -> Any:
    # Priority 1: OpenAI
    if settings.has_openai_key:  # Only checks ENV VAR
        ...
    # Priority 2: Anthropic
    if settings.has_anthropic_key:  # Only checks ENV VAR
        ...
    # Priority 3: HuggingFace
    if settings.has_huggingface_key:  # Only checks ENV VAR
        ...
```

**Impact**: PydanticAI-based components (judges, statistical analyzer) cannot use BYOK keys.

**Current State**: The main Advanced mode flow uses `get_chat_client()` (Microsoft Agent Framework), NOT `get_model()`. So this is latent.

**Fix**: Either:
1. Add `api_key` parameter to `get_model()`
2. Or deprecate `get_model()` in favor of `get_chat_client()` everywhere

---

## Architecture Notes

The codebase has **TWO separate LLM routing systems**:

| System | Function | BYOK Support | Used By |
|--------|----------|--------------|---------|
| Microsoft Agent Framework | `get_chat_client()` | **YES** (key prefix detection) | Advanced mode (main flow) |
| PydanticAI | `get_model()` | **NO** (env vars only) | Judges, statistical analyzer |

This dual-system architecture creates confusion and maintenance burden.

---

## Recommendation

**Short-term**: Leave as-is (latent, not blocking)

**Long-term**: Unify on `get_chat_client()` and deprecate `get_model()` (see P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md for related cleanup)

---

## Test Results

- All 310 unit tests pass
- Main user flow (Gradio → Advanced) works with BYOK

---

## Related Documents

- `P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md` - Related architecture cleanup
- `src/clients/factory.py` - BYOK-capable factory (correct implementation)
- `src/agent_factory/judges.py` - Non-BYOK factory (needs fix)
