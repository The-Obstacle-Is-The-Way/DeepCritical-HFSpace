# P0 - Free Tier Synthesis Incorrectly Uses Server-Side API Keys

**Status:** RESOLVED
**Priority:** P0 (Breaks Free Tier Promise)
**Found:** 2025-11-30
**Resolved:** 2025-11-30
**Component:** `src/orchestrators/simple.py`, `src/agent_factory/judges.py`

## Resolution Summary

The architectural bug where Simple Mode synthesis incorrectly used server-side API keys has been fixed.
We implemented a dedicated `synthesize()` method in `HFInferenceJudgeHandler` that uses the free
HuggingFace Inference API, consistent with the judging phase.

### Fix Details

1.  **New Feature**: Added `synthesize()` method to `HFInferenceJudgeHandler` (and `JudgeHandler` protocol).
    -   Uses `huggingface_hub.InferenceClient.chat_completion` (Free Tier).
    -   Mirrors the `assess()` logic for consistent free access.

2.  **Orchestrator Logic Update**:
    -   `SimpleOrchestrator` now checks `if hasattr(self.judge, "synthesize")`.
    -   If true (Free Tier), it calls `judge.synthesize()` directly, skipping `get_model()`/`pydantic_ai`.
    -   If false (Paid Tier), it falls back to the existing `pydantic_ai` agent flow using `get_model()`.

3.  **Test Coverage**:
    -   Updated `tests/unit/orchestrators/test_simple_synthesis.py` to mock `judge.synthesize`.
    -   Added new test case ensuring Free Tier path is taken when available.
    -   Fixed integration tests to simulate Free Tier correctly.

### Verification

-   **Unit Tests**: `tests/unit/orchestrators/test_simple_synthesis.py` passed (7/7).
-   **Integration**: `tests/integration/test_simple_mode_synthesis.py` passed.
-   **Full Suite**: `make check` passed (310/310 tests).

---

## Symptom (Archive)

When using Simple Mode (Free Tier) without providing a user API key, users see:

```
> ⚠️ **Note**: AI narrative synthesis unavailable. Showing structured summary.
> _Error: OpenAIError_
```

This is confusing because the user didn't configure any OpenAI key - they expected Free Tier to work.

## Root Cause

**Architecture bug: Synthesis is decoupled from JudgeHandler selection.**

| Component | Paid Tier | Free Tier |
|-----------|-----------|-----------|
| Judge | `JudgeHandler` (uses `get_model()`) | `HFInferenceJudgeHandler` (free HF Inference) |
| Synthesis | `get_model()` | **BUG: Also uses `get_model()`** |

**Flow:**
1. User selects Simple mode, leaves API key empty
2. `app.py` correctly creates `HFInferenceJudgeHandler` for judging (works)
3. Search works (no keys needed for PubMed/ClinicalTrials/Europe PMC)
4. Judge works (HFInferenceJudgeHandler uses free HuggingFace inference)
5. **BUG:** Synthesis calls `get_model()` in `simple.py:547`
6. `get_model()` checks `settings.has_openai_key` → reads SERVER-SIDE env vars
7. If ANY server-side key is set (even broken), synthesis tries to use it
8. This VIOLATES the Free Tier promise - user didn't provide a key!

**The bug is NOT about broken keys - it's about synthesis ignoring the Free Tier selection.**

## Impact

- **User Confusion**: User didn't provide a key, sees "OpenAIError"
- **Free Tier Perception**: Makes Free Tier seem broken when it's actually working (template synthesis is still useful)
- **Demo Quality**: Hackathon judges may think the app is broken

## Fix Options

### Option A: Remove/Fix Admin Key (Quick Fix for Hackathon)
Remove or update the `OPENAI_API_KEY` secret on HuggingFace Spaces.
- If removed: Free Tier works as designed (template synthesis)
- If fixed: OpenAI synthesis works

**Pros:** Instant fix, no code changes
**Cons:** Doesn't fix the underlying UX issue

### Option B: Better Error Message
Change error message to be more user-friendly:

```python
# src/orchestrators/simple.py:569-573
error_note = (
    f"\n\n> ⚠️ **Note**: AI narrative synthesis unavailable. "
    f"Showing structured summary.\n"
    f"> _Tip: Provide your own API key for full synthesis._\n"
)
```

**Pros:** Clearer UX
**Cons:** Hides the real error for debugging

### Option C: Provider Fallback Chain (Best Long-term)
If primary provider fails, try next provider before falling back to template:

```python
def get_model_with_fallback() -> Any:
    """Try providers in order, return first that works."""
    from src.utils.exceptions import ConfigurationError

    providers = []
    if settings.has_openai_key:
        providers.append(("openai", lambda: OpenAIChatModel(...)))
    if settings.has_anthropic_key:
        providers.append(("anthropic", lambda: AnthropicModel(...)))
    if settings.has_huggingface_key:
        providers.append(("huggingface", lambda: HuggingFaceModel(...)))

    for name, factory in providers:
        try:
            return factory()
        except Exception as e:
            logger.warning(f"Provider {name} failed: {e}")
            continue

    raise ConfigurationError("No working LLM provider available")
```

**Pros:** Most robust, graceful degradation
**Cons:** More complex, may hide real errors

### Option D: Validate Key Before Using (Recommended)
Add key validation to `get_model()`:

```python
def get_model() -> Any:
    if settings.has_openai_key:
        # Quick validation - check key format
        key = settings.openai_api_key
        if not key or not key.startswith("sk-"):
            logger.warning("Invalid OpenAI key format, trying next provider")
        else:
            return OpenAIChatModel(...)
    # ... continue to next provider
```

**Pros:** Catches obviously invalid keys early
**Cons:** Can't catch quota/permission issues without API call

## Recommended Action (Hackathon)

1. **Immediate**: Remove `OPENAI_API_KEY` from HuggingFace Space secrets, OR replace with valid key
2. **If key is valid**: Check if model `gpt-5` is accessible (may need to use `gpt-4o` instead)

## Test Plan

1. Remove all secrets from HuggingFace Space
2. Run Simple mode query
3. Verify: Search works, Judge works, Synthesis shows template (no error message)

## Related

- `docs/bugs/P0_SYNTHESIS_PROVIDER_MISMATCH.md` (RESOLVED - handles "no keys" case)
- This bug is specifically about "key exists but broken" case
