# P0 Critical Bugs - DeepBoner Demo Broken

**Date**: 2025-11-28
**Status**: ACTIVE - Demo is non-functional
**Priority**: P0 - Blocking hackathon submission

---

## Summary

The Gradio demo is completely non-functional. Both Simple and Advanced modes fail to produce results.

---

## Bug 1: Free Tier LLM Quota Exhausted (P0)

**Symptoms**:
- "Found 20 new sources (0 total)" in UI
- Judge returns 0% confidence
- Loops until max iterations
- Final report shows "Found 0 sources"

**Root Cause**:
HuggingFace Inference API free tier quota is exhausted:
```
402 Client Error: Payment Required
You have exceeded your monthly included credits for Inference Providers
```

All 3 fallback models fail:
1. `meta-llama/Llama-3.1-8B-Instruct` - 402
2. `mistralai/Mistral-7B-Instruct-v0.3` - 402
3. `HuggingFaceH4/zephyr-7b-beta` - 402

**Impact**:
- Free tier users cannot use the demo AT ALL
- Judge always returns "continue" with 0% confidence
- Evidence IS found but never synthesized

**Fix Options**:
1. **Upgrade HF account to PRO** (~$9/month) - immediate fix
2. **Add HF_TOKEN env var** in HF Spaces secrets
3. **Fall back to mock judge** when all LLMs fail (not great UX)
4. **Show clear error message** instead of fake "0 sources"

---

## Bug 2: Evidence Counter Shows 0 After Dedup (P1)

**Symptoms**:
- "Found 20 new sources (0 total)"
- Evidence is found but total is 0

**Root Cause**:
On HuggingFace Spaces, the embeddings service may be failing silently.
The `_deduplicate_and_rank` function returns empty list instead of original.

**Code Location**: `src/orchestrator.py:219`
```python
all_evidence = await self._deduplicate_and_rank(all_evidence, query)
```

If this returns `[]`, we lose all evidence.

**Fix**:
```python
# Add defensive check
deduped = await self._deduplicate_and_rank(all_evidence, query)
if not deduped and all_evidence:
    logger.warning("Deduplication returned empty, keeping original")
    # Keep original evidence
else:
    all_evidence = deduped
```

---

## Bug 3: API Key Not Passed to Advanced Mode (P0)

**Symptoms**:
- User enters OpenAI API key
- Selects Advanced mode
- Gets error or uses wrong/no key

**Root Cause**: CONFIRMED
The user-provided API key is **NEVER passed** to MagenticOrchestrator!

**Code Flow**:
1. `research_agent()` receives `api_key` from Gradio ✅
2. `configure_orchestrator(user_api_key=api_key)` is called ✅
3. For Simple mode: `JudgeHandler(model=OpenAIModel(..., api_key=user_api_key))` ✅
4. For Advanced mode: `MagenticOrchestrator(max_rounds=...)` - **NO API KEY PASSED** ❌

**Bug Location 1**: `src/orchestrator_factory.py:48-52`
```python
if effective_mode == "advanced":
    orchestrator_cls = _get_magentic_orchestrator_class()
    return orchestrator_cls(
        max_rounds=config.max_iterations if config else 10,
        # MISSING: api_key or chat_client parameter!
    )
```

**Bug Location 2**: `src/agents/magentic_agents.py:24-27`
```python
client = chat_client or OpenAIChatClient(
    model_id=settings.openai_model,
    api_key=settings.openai_api_key,  # READS FROM ENV, NOT USER INPUT!
)
```

**Fix Required**:
1. Pass `user_api_key` to `create_orchestrator()`
2. Create `OpenAIChatClient` with user's key
3. Pass `chat_client` to `MagenticOrchestrator`
4. Propagate to all agent factories

---

## Bug 4: Singleton EmbeddingService Causes Cross-Session Pollution (P0)

**Symptoms**:
- First query: "Found 20 new sources (20 total)" ✅
- Second query: "Found 20 new sources (0 total)" ❌
- Same query twice: 0 sources second time

**Root Cause**: CONFIRMED
The EmbeddingService is a **SINGLETON** that persists across ALL Gradio requests!

**Code Location**: `src/services/embeddings.py:164-172`
```python
_embedding_service: EmbeddingService | None = None  # SINGLETON - NEVER RESET!

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()  # Created ONCE per process
    return _embedding_service
```

**What Happens**:
1. Query 1: Finds 20 articles → adds to ChromaDB → `unique = 20`
2. Query 2: Finds 20 articles → `search_similar()` matches Query 1's data → `is_duplicate=True` → `unique = 0`
3. Evidence list becomes empty after deduplication!

**The Real Bug**: `_deduplicate_and_rank()` returns empty list and REPLACES all_evidence:
```python
all_evidence = await self._deduplicate_and_rank(all_evidence, query)  # Returns []!
```

**Fix Options**:
1. **Clear collection per session**: Add `clear()` method and call at start of each `run()`
2. **Use session-scoped collections**: Create unique collection name per Gradio session
3. **Don't use singleton**: Create fresh EmbeddingService per orchestrator run
4. **Defensive check**: If dedup returns empty but input wasn't, keep original

---

## Verification Commands

```bash
# Test search works
uv run python -c "
import asyncio
from src.tools.pubmed import PubMedTool
async def test():
    tool = PubMedTool()
    results = await tool.search('female libido', 5)
    print(f'Found {len(results)} results')
asyncio.run(test())
"

# Test HF inference (will fail with 402 if quota exhausted)
uv run python -c "
from huggingface_hub import InferenceClient
client = InferenceClient()
resp = client.chat_completion(
    messages=[{'role': 'user', 'content': 'Hi'}],
    model='meta-llama/Llama-3.1-8B-Instruct',
    max_tokens=10
)
print(resp)
"
```

---

## Immediate Actions

### Option A: Add HF Pro Account (Recommended)
1. Upgrade HF account to PRO: https://huggingface.co/pricing
2. Generate access token with "inference" scope
3. Add `HF_TOKEN` secret to HF Spaces
4. Verify in HFInferenceJudgeHandler

### Option B: Require Paid API Key
1. Remove "Free Tier" option from UI
2. Make API key required
3. Update messaging

### Option C: Better Error Handling
1. Detect 402 errors specifically
2. Show user-friendly message: "Free tier exhausted, please add API key"
3. Don't loop - fail fast with clear explanation

---

## Definition of Done

- [ ] Demo works with free tier OR shows clear error
- [ ] Demo works with OpenAI key (Simple + Advanced)
- [ ] Demo works with Anthropic key (Simple only)
- [ ] Evidence is correctly accumulated
- [ ] Final report shows actual sources found
- [ ] No silent failures
