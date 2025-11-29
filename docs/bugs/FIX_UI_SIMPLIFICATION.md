# UI Simplification: Remove API Provider Dropdown

**Issues**: #52, #53
**Priority**: P1 - UX improvement for hackathon demo
**Estimated Time**: 30 minutes
**Senior Review**: ✅ Approved with changes (incorporated below)

---

## Problem

The current UI has confusing BYOK (Bring Your Own Key) settings:

1. **Provider dropdown is misleading** - Shows "openai" but actually uses free tier when no key
2. **Examples table shows useless columns** - API Key (empty), Provider (ignored)
3. **Anthropic doesn't work with Advanced mode** - Only OpenAI has `agent-framework` support

## Solution

Remove `api_provider` dropdown entirely. Auto-detect provider from key prefix.

**Functionality preserved:**
- Simple mode: Free tier, OpenAI, OR Anthropic (all work)
- Advanced mode: OpenAI only (Magentic multi-agent requires `OpenAIChatClient`)

---

## Implementation

### File: `src/app.py`

#### Change 1: Update `configure_orchestrator()` signature (lines 23-28)

```python
# BEFORE
def configure_orchestrator(
    use_mock: bool = False,
    mode: str = "simple",
    user_api_key: str | None = None,
    api_provider: str = "openai",  # ← REMOVE
) -> tuple[Any, str]:

# AFTER
def configure_orchestrator(
    use_mock: bool = False,
    mode: str = "simple",
    user_api_key: str | None = None,
) -> tuple[Any, str]:
```

#### Change 2: Update docstring (lines 29-40)

```python
# AFTER
    """
    Create an orchestrator instance.

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode ("simple" or "advanced")
        user_api_key: Optional user-provided API key (BYOK) - auto-detects provider

    Returns:
        Tuple of (Orchestrator instance, backend_name)
    """
```

#### Change 3: Replace provider logic with auto-detection (lines 62-88)

```python
# BEFORE (lines 62-88) - complex provider checking with api_provider param

# AFTER - auto-detect from key prefix
    # 2. Paid API Key (User provided or Env)
    elif user_api_key and user_api_key.strip():
        # Auto-detect provider from key prefix
        model: AnthropicModel | OpenAIModel
        if user_api_key.startswith("sk-ant-"):
            # Anthropic key
            anthropic_provider = AnthropicProvider(api_key=user_api_key)
            model = AnthropicModel(settings.anthropic_model, provider=anthropic_provider)
            backend_info = "Paid API (Anthropic)"
        elif user_api_key.startswith("sk-"):
            # OpenAI key
            openai_provider = OpenAIProvider(api_key=user_api_key)
            model = OpenAIModel(settings.openai_model, provider=openai_provider)
            backend_info = "Paid API (OpenAI)"
        else:
            raise ValueError(
                "Invalid API key format. Expected sk-... (OpenAI) or sk-ant-... (Anthropic)"
            )
        judge_handler = JudgeHandler(model=model)

    # 3. Environment API Keys (fallback)
    elif os.getenv("OPENAI_API_KEY"):
        judge_handler = JudgeHandler(model=None)  # Uses env key
        backend_info = "Paid API (OpenAI from env)"

    elif os.getenv("ANTHROPIC_API_KEY"):
        judge_handler = JudgeHandler(model=None)  # Uses env key
        backend_info = "Paid API (Anthropic from env)"

    # 4. Free Tier (HuggingFace Inference)
    else:
        judge_handler = HFInferenceJudgeHandler()
        backend_info = "Free Tier (Llama 3.1 / Mistral)"
```

#### Change 4: Update `research_agent()` signature (lines 105-111)

```python
# BEFORE
async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    mode: str = "simple",
    api_key: str = "",
    api_provider: str = "openai",  # ← REMOVE
) -> AsyncGenerator[str, None]:

# AFTER
async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    mode: str = "simple",
    api_key: str = "",
) -> AsyncGenerator[str, None]:
```

#### Change 5: Update docstring (lines 112-124)

```python
# AFTER
    """
    Gradio chat function that runs the research agent.

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        mode: Orchestrator mode ("simple" or "advanced")
        api_key: Optional user-provided API key (BYOK - auto-detects provider)

    Yields:
        Markdown-formatted responses for streaming
    """
```

#### Change 6: Fix Advanced mode check (line 139)

```python
# BEFORE
if mode == "advanced" and not (has_openai or (has_user_key and api_provider == "openai")):

# AFTER - auto-detect OpenAI key from prefix
is_openai_user_key = user_api_key and user_api_key.startswith("sk-") and not user_api_key.startswith("sk-ant-")
if mode == "advanced" and not (has_openai or is_openai_user_key):
    yield (
        "⚠️ **Advanced mode requires OpenAI API key.** "
        "Anthropic keys only work in Simple mode. Falling back to Simple.\n\n"
    )
    mode = "simple"
```

#### Change 7: Remove premature "Using your key" message (lines 146-151)

```python
# BEFORE - uses api_provider which no longer exists
if has_user_key:
    yield (
        f"🔑 **Using your {api_provider.upper()} API key** - "
        "Your key is used only for this session and is never stored.\n\n"
    )

# AFTER - remove this block entirely
# The backend_name from configure_orchestrator already shows "Paid API (OpenAI)" or "Paid API (Anthropic)"
# No need for duplicate messaging
```

#### Change 8: Update configure_orchestrator call (lines 165-170)

```python
# BEFORE
orchestrator, backend_name = configure_orchestrator(
    use_mock=False,
    mode=mode,
    user_api_key=user_api_key,
    api_provider=api_provider,  # ← REMOVE
)

# AFTER
orchestrator, backend_name = configure_orchestrator(
    use_mock=False,
    mode=mode,
    user_api_key=user_api_key,
)
```

#### Change 9: Simplify examples (lines 210-229)

```python
# BEFORE - 4 items per example
examples=[
    ["What drugs improve female libido post-menopause?", "simple", "", "openai"],
    ["Clinical trials for erectile dysfunction alternatives to PDE5 inhibitors?", "simple", "", "openai"],
    ["Evidence for testosterone therapy in women with HSDD?", "simple", "", "openai"],
],

# AFTER - 2 items per example (query, mode) - API key always empty in examples
examples=[
    ["What drugs improve female libido post-menopause?", "simple"],
    ["Clinical trials for ED alternatives to PDE5 inhibitors?", "simple"],
    ["Evidence for testosterone therapy in women with HSDD?", "simple"],
],
```

#### Change 10: Update additional_inputs (lines 231-252)

```python
# BEFORE - 3 inputs (mode, api_key, api_provider)
additional_inputs=[
    gr.Radio(
        choices=["simple", "advanced"],
        value="simple",
        label="Orchestrator Mode",
        info="Simple: Linear (Free Tier Friendly) | Advanced: Multi-Agent (Requires OpenAI)",
    ),
    gr.Textbox(
        label="🔑 API Key (Optional - BYOK)",
        placeholder="sk-... or sk-ant-...",
        type="password",
        info="Enter your own API key. Never stored.",
    ),
    gr.Radio(  # ← REMOVE THIS ENTIRE BLOCK
        choices=["openai", "anthropic"],
        value="openai",
        label="API Provider",
        info="Select the provider for your API key",
    ),
],

# AFTER - 2 inputs (mode, api_key)
additional_inputs=[
    gr.Radio(
        choices=["simple", "advanced"],
        value="simple",
        label="Orchestrator Mode",
        info="Simple: Works with any key or free tier | Advanced: Requires OpenAI key",
    ),
    gr.Textbox(
        label="🔑 API Key (Optional)",
        placeholder="sk-... (OpenAI) or sk-ant-... (Anthropic)",
        type="password",
        info="Leave empty for free tier. Auto-detects provider from key prefix.",
    ),
],
```

#### Change 11: Update accordion label (line 230)

```python
# BEFORE
additional_inputs_accordion=gr.Accordion(label="⚙️ Settings", open=False),

# AFTER
additional_inputs_accordion=gr.Accordion(label="⚙️ Settings (Free tier works without API key)", open=False),
```

---

## Testing Checklist

### Manual Tests
- [ ] **No key**: Shows "Free Tier (Llama 3.1 / Mistral)" in backend
- [ ] **OpenAI key (sk-...)**: Shows "Paid API (OpenAI)" in backend
- [ ] **Anthropic key (sk-ant-...)**: Shows "Paid API (Anthropic)" in backend
- [ ] **Invalid key format**: Shows error message
- [ ] **Anthropic key + Advanced mode**: Falls back to Simple with warning
- [ ] **OpenAI key + Advanced mode**: Uses full Magentic multi-agent
- [ ] **Examples table**: Shows only 2 columns (query, mode)
- [ ] **MCP server**: Still accessible at `/gradio_api/mcp/`

### Unit Test Updates
- [ ] `tests/unit/test_app_smoke.py` - may need update if checking input count

---

## Definition of Done

- [ ] `api_provider` parameter removed from `configure_orchestrator()`
- [ ] `api_provider` parameter removed from `research_agent()`
- [ ] Auto-detection logic works for `sk-` and `sk-ant-` prefixes
- [ ] Advanced mode check uses auto-detection (not removed param)
- [ ] "Using your X key" message removed (backend_name handles this)
- [ ] Examples table shows 2 columns
- [ ] Accordion label updated
- [ ] Placeholder text shows both key formats
- [ ] All existing tests pass
- [ ] MCP server still works

---

## Mode Compatibility Matrix (Unchanged)

| Mode | No Key | OpenAI Key | Anthropic Key |
|------|--------|------------|---------------|
| **Simple** | ✅ Free tier | ✅ GPT-5.1 | ✅ Claude Sonnet 4.5 |
| **Advanced** | ⚠️ Falls back | ✅ Full Magentic | ⚠️ Falls back to Simple |

---

## Related
- Issue #52: UI Polish - Examples table confusion
- Issue #53: API Provider Simplification
- Senior Review: Approved 2025-11-28
