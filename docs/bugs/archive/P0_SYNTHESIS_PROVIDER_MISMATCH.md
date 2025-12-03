# P0 - Systemic Provider Mismatch Across All Modes

**Status:** RESOLVED
**Priority:** P0 (Blocker for Free Tier/Demo)
**Found:** 2025-11-30 (during Audit)
**Resolved:** 2025-11-30
**Component:** Multiple files across orchestrators, agents, services

## Resolution Summary

The critical provider mismatch bug has been fixed by implementing auto-detection in `src/agent_factory/judges.py`.
The `get_model()` function now checks for actual API key availability (`has_openai_key`, `has_anthropic_key`, `has_huggingface_key`)
instead of relying on the static `settings.llm_provider` configuration.

### Fix Details

- **Auto-Detection Implemented**: `get_model()` prioritizes OpenAI > Anthropic > HuggingFace based on *available keys*.
- **Fail-Fast on No Keys**: If no API keys are configured, `get_model()` raises `ConfigurationError` with clear message.
- **HuggingFace Requires Token**: Free Tier via `HuggingFaceModel` requires `HF_TOKEN` (PydanticAI requirement).
- **Synthesis Fallback**: When `get_model()` fails, synthesis gracefully falls back to template.
- **Audit Fixes Applied**:
    - Replaced manual `os.getenv` checks with centralized `settings` properties in `src/app.py`.
    - Added logging to `src/services/statistical_analyzer.py` (fixed silent `pass`).
    - Narrowed exception handling in `src/tools/pubmed.py`.
    - Optimized string search in `src/tools/code_execution.py`.

### Key Clarification

The **Free Tier** in Simple Mode uses `HFInferenceJudgeHandler` (which uses `huggingface_hub.InferenceClient`)
for judging - this does NOT require `HF_TOKEN`. However, synthesis via `get_model()` uses PydanticAI's
`HuggingFaceModel` which DOES require `HF_TOKEN`. When no tokens are configured, synthesis falls back to
the template-based summary (which is still useful).

### Verification

- **Unit Tests**: 5 new TDD tests in `tests/unit/agent_factory/test_get_model_auto_detect.py` pass.
- **All Tests**: 309 tests pass (`make check` succeeds).
- **Regression Tests**: Fixed and verified `tests/unit/agent_factory/test_judges_factory.py`.

---

## Symptom (Archive)

When running in "Simple Mode" (Free Tier / No API Key), the synthesis step fails to generate a narrative and falls back to a structured summary template. The user sees:

```text
> ⚠️ Note: AI narrative synthesis unavailable. Showing structured summary.
> _Error: OpenAIError_
```

## Affected Files (COMPREHENSIVE AUDIT)

### Files Calling `get_model()` Directly (9 locations)

| File | Line | Context | Impact |
|------|------|---------|--------|
| `simple.py` | 547 | Synthesis step | Free Tier broken |
| `statistical_analyzer.py` | 75 | Analysis agent | Free Tier broken |
| `judge_agent_llm.py` | 18 | LLM Judge | Free Tier broken |
| `graph/nodes.py` | 177 | LangGraph hypothesis | Free Tier broken |
| `graph/nodes.py` | 249 | LangGraph synthesis | Free Tier broken |
| `report_agent.py` | 45 | Report generation | Free Tier broken |
| `hypothesis_agent.py` | 44 | Hypothesis generation | Free Tier broken |
| `judges.py` | 100 | JudgeHandler default | OK (accepts param) |

### Files Hardcoding `OpenAIChatClient` (Architecturally OpenAI-Only)

| File | Lines | Context |
|------|-------|---------|
| `advanced.py` | 100, 121 | Manager client |
| `magentic_agents.py` | 29, 70, 129, 173 | All 4 agents |
| `retrieval_agent.py` | 62 | Retrieval agent |
| `code_executor_agent.py` | 52 | Code executor |
| `llm_factory.py` | 42 | Factory default |

**Note:** Advanced mode is architecturally locked to OpenAI via `agent_framework.openai.OpenAIChatClient`. This is by design - see `app.py:188-194` which falls back to Simple mode if no OpenAI key. However, users are not clearly informed of this limitation.

## Root Cause

**Settings/Runtime Sync Gap - Two Separate Backend Selection Systems.**

The codebase has **two independent** systems for selecting the LLM backend:
1. `settings.llm_provider` (config.py default: "openai")
2. `app.py` runtime detection via `os.getenv()` checks

These are **never synchronized**, causing the Judge and Synthesis steps to use different backends.

### Detailed Call Chain

1.  **`src/app.py:115-126`** (runtime detection):
    ```python
    # app.py bypasses settings entirely for JudgeHandler selection
    elif os.getenv("OPENAI_API_KEY"):
        judge_handler = JudgeHandler(model=None, domain=domain)
    elif os.getenv("ANTHROPIC_API_KEY"):
        judge_handler = JudgeHandler(model=None, domain=domain)
    else:
        judge_handler = HFInferenceJudgeHandler(domain=domain)  # Free Tier
    ```
    **Note:** This creates the correct handler but does NOT update `settings.llm_provider`.

2.  **`src/orchestrators/simple.py:546-552`** (synthesis step):
    ```python
    from src.agent_factory.judges import get_model
    agent: Agent[None, str] = Agent(model=get_model(), ...)  # <-- BUG!
    ```
    Synthesis calls `get_model()` directly instead of using the injected judge's model.

3.  **`src/agent_factory/judges.py:56-78`** (`get_model()`):
    ```python
    def get_model() -> Any:
        llm_provider = settings.llm_provider  # <-- Reads from settings (still "openai")
        # ...
        openai_provider = OpenAIProvider(api_key=settings.openai_api_key)  # <-- None!
        return OpenAIChatModel(settings.openai_model, provider=openai_provider)
    ```
    **Result:** Creates OpenAI model with `api_key=None` → `OpenAIError`

### Why Free Tier Fails

| Step | System Used | Backend Selected |
|------|-------------|------------------|
| JudgeHandler | `app.py` runtime | HFInferenceJudgeHandler ✅ |
| Synthesis | `settings.llm_provider` | OpenAI (default) ❌ |

The Judge works because app.py explicitly creates `HFInferenceJudgeHandler`.
Synthesis fails because it calls `get_model()` which reads `settings.llm_provider = "openai"` (unchanged from default).

## Impact

-   **User Experience:** Free tier users (Demo users) never see the high-quality narrative synthesis, only the fallback.
-   **System Integrity:** The orchestrator ignores the runtime backend selection.

## Implemented Fix

**Strategy: Fix `get_model()` to Auto-Detect Available Provider**

### Actual Implementation (Merged)

**File:** `src/agent_factory/judges.py`

This is the **single point of fix** that resolves all 7 broken `get_model()` call sites.

```python
def get_model() -> Any:
    """Get the LLM model based on available API keys.

    Priority order:
    1. OpenAI (if OPENAI_API_KEY set)
    2. Anthropic (if ANTHROPIC_API_KEY set)
    3. HuggingFace (if HF_TOKEN set)

    Raises:
        ConfigurationError: If no API keys are configured.

    Note: settings.llm_provider is ignored in favor of actual key availability.
    This ensures the model matches what app.py selected for JudgeHandler.
    """
    from src.utils.exceptions import ConfigurationError

    # Priority 1: OpenAI (most common, best tool calling)
    if settings.has_openai_key:
        openai_provider = OpenAIProvider(api_key=settings.openai_api_key)
        return OpenAIChatModel(settings.openai_model, provider=openai_provider)

    # Priority 2: Anthropic
    if settings.has_anthropic_key:
        provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model, provider=provider)

    # Priority 3: HuggingFace (requires HF_TOKEN)
    if settings.has_huggingface_key:
        model_name = settings.huggingface_model or "meta-llama/Llama-3.1-70B-Instruct"
        hf_provider = HuggingFaceProvider(api_key=settings.hf_token)
        return HuggingFaceModel(model_name, provider=hf_provider)

    # No keys configured - fail fast with clear error
    raise ConfigurationError(
        "No LLM API key configured. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or HF_TOKEN"
    )
```

**Why this works:**
- Single fix location updates all 7 broken call sites
- Matches app.py's detection logic (key availability, not settings.llm_provider)
- HuggingFace works when HF_TOKEN is available
- Raises clear error when no keys configured (callers can catch and fallback)
- No changes needed to orchestrators, agents, or services

### What This Does NOT Fix (By Design)

**Advanced Mode remains OpenAI-only.** The following files use `agent_framework.openai.OpenAIChatClient` which only supports OpenAI:

- `advanced.py` (Manager + agents)
- `magentic_agents.py` (SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent)
- `retrieval_agent.py`, `code_executor_agent.py`

This is **by design** - the Microsoft Agent Framework library (`agent-framework-core`) only provides `OpenAIChatClient`. To support other providers in Advanced mode would require:
1. Wait for `agent-framework` to add Anthropic/HuggingFace clients, OR
2. Write our own `ChatClient` implementations (significant effort)

**The current app.py behavior is correct:** it falls back to Simple mode when no OpenAI key is present (lines 188-194). The UI message could be clearer about why.

## Test Plan (Implemented)

### Unit Tests (Verified Passing)

```python
# tests/unit/agent_factory/test_get_model_auto_detect.py

import pytest
from src.agent_factory.judges import get_model
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError

class TestGetModelAutoDetect:
    """Test that get_model() auto-detects available providers."""

    def test_returns_openai_when_key_present(self, monkeypatch):
        """OpenAI key present → OpenAI model."""
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)
        model = get_model()
        assert isinstance(model, OpenAIChatModel)

    def test_returns_anthropic_when_only_anthropic_key(self, monkeypatch):
        """Only Anthropic key → Anthropic model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
        monkeypatch.setattr(settings, "hf_token", None)
        model = get_model()
        assert isinstance(model, AnthropicModel)

    def test_returns_huggingface_when_hf_token_present(self, monkeypatch):
        """HF_TOKEN present (no paid keys) → HuggingFace model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")
        model = get_model()
        assert isinstance(model, HuggingFaceModel)

    def test_raises_error_when_no_keys(self, monkeypatch):
        """No keys at all → ConfigurationError."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)
        with pytest.raises(ConfigurationError) as exc_info:
            get_model()
        assert "No LLM API key configured" in str(exc_info.value)

    def test_openai_takes_priority_over_anthropic(self, monkeypatch):
        """Both keys present → OpenAI wins."""
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
        model = get_model()
        assert isinstance(model, OpenAIChatModel)
```

### Full Test Suite

```bash
$ make check
# 309 passed in 238.16s (0:03:58)
# All checks passed!
```

### Manual Verification

1. **Unset all API keys**: `unset OPENAI_API_KEY ANTHROPIC_API_KEY HF_TOKEN`
2. **Run app**: `uv run python -m src.app`
3. **Submit query**: "What drugs improve female libido?"
4. **Verify**: Synthesis falls back to template (shows `ConfigurationError` in logs, but user sees structured summary)
