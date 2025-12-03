# P1 BUG: HuggingFace Router 500 Error via Novita Provider

**Status**: ACTIVE - Upstream Infrastructure Issue
**Priority**: P1 (Free Tier Broken)
**Discovered**: 2025-12-02
**Related**: CLAUDE.md (Llama/Hyperbolic issue)

---

## Symptom

```
❌ **ERROR**: Workflow error: 500 Server Error: Internal Server Error for url:
https://router.huggingface.co/novita/v3/openai/chat/completions
```

Free tier users (no API key) cannot use the system.

---

## Stack Trace

```text
User (no API key)
    ↓
src/clients/factory.py:get_chat_client()
    ↓
src/clients/huggingface.py:HuggingFaceChatClient
    ↓
Model: Qwen/Qwen2.5-72B-Instruct (from config.py)
    ↓
huggingface_hub.InferenceClient
    ↓
HuggingFace Router: router.huggingface.co
    ↓
Routes to: NOVITA (third-party inference provider)
    ↓
❌ Novita returns 500 Internal Server Error
```

---

## Root Cause

**HuggingFace doesn't host all models directly.** For some models, they route to third-party inference providers:

| Model | Provider | Status |
|-------|----------|--------|
| Llama-3.1-70B | Hyperbolic | ❌ "staging mode" auth issues |
| Qwen2.5-72B | Novita | ❌ 500 Internal Server Error |

We switched from Llama to Qwen specifically to avoid Hyperbolic's issues. Now Novita is having its own problems.

**This is an upstream infrastructure issue - not a bug in our code.**

---

## Evidence

From the error URL:
```
https://router.huggingface.co/novita/v3/openai/chat/completions
                              ^^^^^^
                              Third-party provider in URL path
```

---

## Potential Fixes

### Option 1: Try a Different Model (Quick)
Find a model that HuggingFace hosts natively (not routed to partners):

```python
# Candidates to test:
# - mistralai/Mistral-7B-Instruct-v0.3
# - microsoft/Phi-3-mini-4k-instruct
# - google/gemma-2-9b-it
```

### Option 2: Add Fallback Logic (Robust)
```python
FALLBACK_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

async def get_response_with_fallback(...):
    for model in FALLBACK_MODELS:
        try:
            return await client.chat_completion(model=model, ...)
        except HfHubHTTPError as e:
            if e.status_code == 500:
                continue
            raise
    raise AllModelsFailedError()
```

### Option 3: Wait for Novita Fix (Passive)
500 errors are typically transient. Novita may fix their infrastructure.

---

## Verification

To check if issue is resolved:
```bash
curl -X POST "https://router.huggingface.co/novita/v3/openai/chat/completions" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

---

## Historical Context

From `CLAUDE.md`:
```
- **HuggingFace (Free Tier):** `Qwen/Qwen2.5-72B-Instruct`
  - Changed from Llama-3.1-70B (Dec 2025) due to HuggingFace routing Llama
    to Hyperbolic provider which has unreliable "staging mode" auth.
```

Now Qwen is being routed to Novita, continuing the pattern of unreliable third-party routing.

---

## Recommendation

**Short-term**: Switch to a model hosted natively by HuggingFace (test candidates above)
**Long-term**: Implement fallback model logic to handle provider outages gracefully
