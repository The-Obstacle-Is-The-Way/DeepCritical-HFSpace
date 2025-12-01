# P1 Bug: HuggingFace Router 401 Unauthorized (Hyperbolic Provider)

**Severity**: P1 (High) - Free Tier completely broken
**Status**: Open
**Discovered**: 2025-12-01
**Reporter**: Production user via HuggingFace Spaces

## Symptom

```
401 Client Error: Unauthorized for url:
https://router.huggingface.co/hyperbolic/v1/chat/completions
Invalid username or password.
```

## Root Cause Analysis

### What Changed (NOT our code)

HuggingFace has migrated their Inference API infrastructure:

1. **Old endpoint** (deprecated): `https://api-inference.huggingface.co`
2. **New endpoint**: `https://router.huggingface.co/{provider}/v1/chat/completions`

The new "router" system routes requests to **partner providers** based on the model:
- `meta-llama/Llama-3.1-70B-Instruct` → **Hyperbolic** (partner)
- Other models → various providers

**Critical Issue**: Hyperbolic requires authentication even for models that were previously "free tier" on HuggingFace's native infrastructure.

### Call Stack Trace

```
User Query (HuggingFace Spaces)
    ↓
src/app.py:research_agent()
    ↓
src/orchestrators/advanced.py:AdvancedOrchestrator.run()
    ↓
src/clients/factory.py:get_chat_client()  [line 69-76]
    → No OpenAI key → Falls back to HuggingFace
    ↓
src/clients/huggingface.py:HuggingFaceChatClient.__init__()  [line 52-56]
    → InferenceClient(model="meta-llama/Llama-3.1-70B-Instruct", token=None)
    ↓
huggingface_hub.InferenceClient.chat_completion()
    → Routes to: https://router.huggingface.co/hyperbolic/v1/chat/completions
    → 401 Unauthorized (Hyperbolic rejects unauthenticated requests)
```

### Evidence

- **huggingface_hub version**: 0.36.0 (latest)
- **pyproject.toml constraint**: `>=0.24.0`
- **HuggingFace Forum Reference**: [API endpoint migration thread](https://discuss.huggingface.co/t/error-https-api-inference-huggingface-co-is-no-longer-supported-please-use-https-router-huggingface-co-hf-inference-instead/169870)

## Impact

| Component | Impact |
|-----------|--------|
| Free Tier (no API key) | **COMPLETELY BROKEN** |
| HuggingFace Spaces demo | **BROKEN** |
| Users without OpenAI key | **Cannot use app** |
| Paid tier (OpenAI key) | Unaffected |

## Proposed Solutions

### Option 1: Switch to Smaller Free Model (Quick Fix)

Change default model from `meta-llama/Llama-3.1-70B-Instruct` to a model that's still hosted on HuggingFace's native infrastructure:

```python
# src/utils/config.py
huggingface_model: str | None = Field(
    default="mistralai/Mistral-7B-Instruct-v0.3",  # Still on HF native
    description="HuggingFace model name"
)
```

**Candidates** (need testing):
- `mistralai/Mistral-7B-Instruct-v0.3`
- `HuggingFaceH4/zephyr-7b-beta`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2-9b-it`

**Pros**: Quick fix, no auth required
**Cons**: Lower quality output than Llama 3.1 70B

### Option 2: Require HF_TOKEN for Free Tier

Document that `HF_TOKEN` is now **required** (not optional) for Free Tier:

```python
# src/clients/factory.py
if not settings.hf_token:
    raise ConfigurationError(
        "HF_TOKEN is now required for HuggingFace free tier. "
        "Get yours at https://huggingface.co/settings/tokens"
    )
```

**Pros**: Keeps Llama 3.1 70B quality
**Cons**: Friction for users, not truly "free" anymore

### Option 3: Server-Side HF_TOKEN on Spaces

Set `HF_TOKEN` as a secret in HuggingFace Spaces settings:
1. Go to Space Settings → Repository Secrets
2. Add `HF_TOKEN` with a valid token
3. Users get free tier without needing their own token

**Pros**: Best UX, transparent to users
**Cons**: Token usage counted against our account

### Option 4: Hybrid Fallback Chain

Try multiple models in order until one works:

```python
FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",  # Best quality (needs token)
    "mistralai/Mistral-7B-Instruct-v0.3",  # Good quality (free)
    "microsoft/Phi-3-mini-4k-instruct",    # Lightweight (free)
]
```

**Pros**: Graceful degradation
**Cons**: Complexity, inconsistent output quality

## Recommended Fix

**Short-term (P1)**: Option 3 - Add `HF_TOKEN` to HuggingFace Spaces secrets

**Long-term**: Option 4 - Implement fallback chain with clear user feedback about which model is active

## Testing

```bash
# Test without token (should fail currently)
unset HF_TOKEN
uv run python -c "
from huggingface_hub import InferenceClient
client = InferenceClient(model='meta-llama/Llama-3.1-70B-Instruct')
response = client.chat_completion(messages=[{'role': 'user', 'content': 'Hi'}])
print(response)
"

# Test with token (should work)
export HF_TOKEN=hf_xxxxx
uv run python -c "
from huggingface_hub import InferenceClient
client = InferenceClient(model='meta-llama/Llama-3.1-70B-Instruct', token='$HF_TOKEN')
response = client.chat_completion(messages=[{'role': 'user', 'content': 'Hi'}])
print(response)
"
```

## References

- [HuggingFace API Migration Thread](https://discuss.huggingface.co/t/error-https-api-inference-huggingface-co-is-no-longer-supported-please-use-https-router-huggingface-co-hf-inference-instead/169870)
- [GitHub Issue: 401 Unauthorized](https://github.com/huggingface/transformers/issues/38289)
- [HuggingFace Inference Endpoints Docs](https://huggingface.co/docs/huggingface_hub/guides/inference)
## Update 2025-12-01 21:45 PST

**Attempted Fix 1**: Switched model from `meta-llama/Llama-3.1-70B-Instruct` (Hyperbolic) to `Qwen/Qwen2.5-72B-Instruct` (routed to **Novita**).

**Result**: Failed with same 401 error on Novita.
```
401 Client Error: Unauthorized for url: https://router.huggingface.co/novita/v3/openai/chat/completions
Invalid username or password.
```

**New Findings**:
1. **All Large Models are Partners**: Both Llama-70B and Qwen-72B are routed to partner providers (Hyperbolic, Novita).
2. **Partners Require Auth**: Partner providers strictly require authentication. Anonymous access is blocked.
3. **Token Propagation Failure**: Even with `HF_TOKEN` set in Spaces secrets, the `huggingface_hub` library might not be picking it up via Pydantic settings if `alias` resolution is flaky in the environment.
4. **Possible Token Permission Issue**: The user's token might lack permissions for Partner Inference endpoints.

**Corrective Actions**:
1. **Robust Config Loading**: Modified `src/utils/config.py` to use `default_factory=lambda: os.environ.get("HF_TOKEN")` to guarantee environment variable reading.
2. **Debug Logging**: Added explicit logging in `src/clients/huggingface.py` to confirming if a token is being used (masked).
3. **Retain Qwen**: Keeping `Qwen/Qwen2.5-72B-Instruct` as it's a capable model. If auth is fixed, it should work.

**Next Steps**:
- Deploy these changes to debug the token loading.
- If token is loaded but still failing, the user must generate a new `HF_TOKEN` with **"Make calls to inference endpoints"** permissions.