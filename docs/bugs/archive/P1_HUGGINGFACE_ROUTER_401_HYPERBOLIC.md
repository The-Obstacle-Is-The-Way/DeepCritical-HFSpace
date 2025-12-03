# P1 Bug: HuggingFace Router 401 Unauthorized

**Severity**: P1 (High)
**Status**: RESOLVED
**Discovered**: 2025-12-01
**Resolved**: 2025-12-01
**Reporter**: Production user via HuggingFace Spaces

## Symptom

```
401 Client Error: Unauthorized for url:
https://router.huggingface.co/hyperbolic/v1/chat/completions
Invalid username or password.
```

## Root Cause

**The HF_TOKEN in `.env` and HuggingFace Spaces secrets was invalid/expired.**

Token `hf_ssayg...` failed `HfApi().whoami()` verification.

## Resolution

1. Generated new HF_TOKEN at https://huggingface.co/settings/tokens
2. Updated `.env` with new token: `hf_gZVBI...`
3. Updated HuggingFace Spaces secret with same token
4. Switched default model from `meta-llama/Llama-3.1-70B-Instruct` to `Qwen/Qwen2.5-72B-Instruct` (better reliability via HF router)

## Verification

```bash
uv run python -c "
import os
from huggingface_hub import InferenceClient, HfApi

token = os.environ['HF_TOKEN']  # Your valid token from .env
api = HfApi(token=token)
print(f'Token valid: {api.whoami()[\"name\"]}')

client = InferenceClient(model='Qwen/Qwen2.5-72B-Instruct', token=token)
response = client.chat_completion(messages=[{'role': 'user', 'content': '2+2=?'}], max_tokens=10)
print(f'Inference works: {response.choices[0].message.content}')
"
# Output:
# Token valid: VibecoderMcSwaggins
# Inference works: 4
```

## Lessons Learned

1. **First-principles debugging**: Before adding complex "fixes", verify basic assumptions (is the token actually valid?)
2. **Token expiration**: HuggingFace tokens can expire or become invalid. Always verify with `whoami()`.
3. **Model routing**: HuggingFace routes large models to partner providers (Hyperbolic, Novita). All require valid auth.

## Files Changed

- `src/utils/config.py`: Changed default model to `Qwen/Qwen2.5-72B-Instruct`
- `src/clients/huggingface.py`: Updated fallback model reference
- `src/agent_factory/judges.py`: Updated fallback model reference
- `src/orchestrators/langgraph_orchestrator.py`: Updated hardcoded model
- `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`: Updated documentation
