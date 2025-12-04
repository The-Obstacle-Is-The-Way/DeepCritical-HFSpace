# P1 Bug: Free Tier Tool Execution Failure

**Date**: 2025-12-03
**Status**: FIXED (PR fix/P1-free-tier-tool-execution)
**Severity**: P1 (Critical - Free Tier Completely Broken)
**Component**: HuggingFaceChatClient + Together.ai Routing + Tool Calling
**Resolution**: Removed premature `__function_invoking_chat_client__ = True` marker from class body

---

## Executive Summary

The Free Tier (HuggingFace) is fundamentally broken due to **multiple interacting issues** that cause tool calls to fail, resulting in garbage output, hallucinated results, and raw JSON appearing in the UI.

**This is NOT a simple 7B model issue** - it's a chain of infrastructure and code problems.

---

## Symptoms

Users on Free Tier see:

1. **Garbage tokens**: "oleon", "UrlParser", "MemoryWarning", "PostalCodes"
2. **Raw tool call XML tags**: `<tool_call>`, `</tool_call>` appearing as text
3. **Raw JSON tool calls**: `{"name": "search_pubmed", "arguments": {...}}`
4. **Hallucinated tool results**: Fake JSON responses that were never returned by actual tools:
   ```json
   {"response": "[{'title': 'Effect of Flibanserin...', ...}]"}
   ```
5. **No actual database searches**: PubMed, ClinicalTrials.gov never queried

---

## Root Cause Analysis

### Cause 1: Model Routed to Third-Party Provider (Together.ai)

**Discovery**: Qwen2.5-7B-Instruct is NOT served by native HuggingFace infrastructure.

```python
# API response from HuggingFace:
{
  "inferenceProviderMapping": {
    "together": {
      "status": "live",
      "providerId": "Qwen/Qwen2.5-7B-Instruct-Turbo"  # <-- TURBO variant!
    },
    "featherless-ai": {
      "status": "live",
      "providerId": "Qwen/Qwen2.5-7B-Instruct"
    }
  }
}
```

**Impact**:
- Native HF-inference returns 404 for this model
- All requests route through Together.ai
- Together serves a "Turbo" variant, not the original
- We cannot control how Together handles tool calling

### Cause 2: Qwen2.5 Uses XML-Style Tool Calling Format

**Discovery**: The model's chat template instructs it to output tool calls in XML format:

```jinja
For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

**Impact**:
- Model outputs `<tool_call>{"name":...}</tool_call>` as **text**
- This text appears in `delta.content` (not `delta.tool_calls`)
- Our streaming code yields this as visible text to the UI
- When tool calling works correctly, the API parses this internally
- When it fails, raw XML appears in output

### Cause 3: Together.ai Turbo Inconsistent Tool Call Parsing

**Discovery**: Together's serving of the Turbo model has inconsistent behavior:

| Test Scenario | Tool Call Behavior |
|---------------|-------------------|
| Simple query, single tool | ✅ Parsed correctly to `tool_calls` |
| Complex multi-agent prompt | ❌ Mixed: some parsed, some as text |
| Multi-turn with tool results | ❌ Model hallucinates fake results |

**Evidence from testing**:
```python
# Simple test - WORKS:
finish_reason: tool_calls
content: None
tool_calls: [ChatCompletionOutputToolCall(function=..., name='search_pubmed')]

# Complex prompt - FAILS:
TEXT[49]: '建档立标'  # Chinese garbage between tool calls
TEXT[X]: '{"name": "search_preprints", ...}'  # Raw JSON as text
```

### Cause 4: Potential Code Bug - Premature Marker Setting

**Discovery**: In `HuggingFaceChatClient`, we set a marker that may prevent tool execution wrapping:

```python
@use_function_invocation   # Decorator checks marker BEFORE wrapping
@use_observability
@use_chat_middleware
class HuggingFaceChatClient(BaseChatClient):
    # This marker causes decorator to return early!
    __function_invoking_chat_client__ = True  # <-- BUG?
```

The `@use_function_invocation` decorator source:
```python
def use_function_invocation(chat_client):
    if getattr(chat_client, FUNCTION_INVOKING_CHAT_CLIENT_MARKER, False):
        return chat_client  # EARLY RETURN - doesn't wrap methods!
    # ... wrapping code never runs ...
```

**Impact**: The decorator sees the marker as `True` and returns early without wrapping `get_response` and `get_streaming_response` with the function invocation handler.

**Status**: NEEDS VERIFICATION - Testing shows methods have `__wrapped__` attribute, suggesting some decoration occurred. May be from other decorators.

### Cause 5: Model Hallucination Under Complexity

**Discovery**: When the model fails to make proper API tool calls, it **simulates** tool use by outputting fake results:

```
{"response": "[{'title': 'Effect of Flibanserin...'}]"}
```

This is pure hallucination - no actual API calls were made. The model is trained to produce tool-like outputs, so when the API tool calling fails, it falls back to text-based simulation.

---

## Verification Steps

### Test 1: Direct InferenceClient (PASSES)

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model='Qwen/Qwen2.5-7B-Instruct')
response = client.chat_completion(
    messages=[{'role': 'user', 'content': 'What is the weather?'}],
    tools=[weather_tool],
    tool_choice='auto',
)
# Result: tool_calls properly parsed, content=None
```

### Test 2: Complex Multi-Agent Prompt (FAILS)

```python
# With our SearchAgent-style prompts:
stream = client.chat_completion(
    messages=[system_prompt, user_query],
    tools=multiple_tools,
    ...
)
# Result: Mix of text content AND tool_calls, garbage tokens appear
```

### Test 3: ChatAgent Single Tool (PARTIAL)

```python
agent = ChatAgent(
    chat_client=HuggingFaceChatClient(),
    tools=[search_pubmed],
    ...
)
result = await agent.run('Search for libido drugs')
# Result: Tool call request made but function NOT executed (tool_calls=0)
```

---

## Impact Assessment

| Aspect | Impact |
|--------|--------|
| Free Tier Users | **100% broken** - Cannot get any useful results |
| Demo Quality | **Unprofessional** - Shows garbage/hallucinations |
| User Trust | **Critical** - Appears completely broken |
| Tool Execution | **Not working** - Tools never actually called |

---

## Fix Options

### Option 1: Remove Premature Marker (QUICK - Test First)

**Location**: `src/clients/huggingface.py:43`

```python
# REMOVE THIS LINE:
__function_invoking_chat_client__ = True
```

Let the `@use_function_invocation` decorator set the marker AFTER wrapping.

**Risk**: Unknown - need to test if this actually enables tool execution.

### Option 2: Switch to Model with Native HF Support

Find a model that runs on native HuggingFace infrastructure (not routed to third parties):

| Model | Size | Native HF? | Tool Calling |
|-------|------|------------|--------------|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ❓ Test | ❓ |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ❓ Test | ✅ |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ❓ Test | Limited |

### Option 3: Simplify Free Tier to Single-Agent

Remove multi-agent complexity for Free Tier:
- Single ChatAgent with simpler prompt
- Direct tool calls instead of MagenticBuilder workflow
- Reduced prompt complexity

### Option 4: Streaming Content Filter (BAND-AID)

Filter garbage from streaming output:

```python
def should_stream_content(text: str) -> bool:
    """Filter garbage from streaming."""
    if text.strip().startswith('{"name":'):
        return False  # Raw tool call JSON
    if '</tool_call>' in text or '<tool_call>' in text:
        return False  # XML tags
    garbage = ["oleon", "UrlParser", "MemoryWarning", "建档立标"]
    if any(g in text for g in garbage):
        return False
    return True
```

**Note**: This hides symptoms but doesn't fix the underlying tool execution failure.

### Option 5: Use Together.ai Directly with Their SDK

Bypass HuggingFace routing entirely:
- Use Together's official SDK
- May have better tool calling support
- Requires new client implementation

---

## Files Involved

| File | Role |
|------|------|
| `src/clients/huggingface.py` | Main HF client - has premature marker |
| `src/clients/factory.py` | Client selection logic |
| `src/agents/magentic_agents.py` | Agent definitions with tools |
| `src/orchestrators/advanced.py` | Multi-agent workflow |
| `src/agents/tools.py` | Tool function definitions |

---

## Recommended Action Plan

### Phase 1: Verify Code Bug (Immediate)

1. Remove `__function_invoking_chat_client__ = True` from HuggingFaceChatClient
2. Test if tool execution now works
3. If yes, verify no regressions with full test suite

### Phase 2: Provider Testing

1. Test which small models have native HF support
2. Evaluate Together.ai direct integration
3. Document provider routing for all candidate models

### Phase 3: Architecture Decision

Based on Phase 1-2 results:
- If code fix works: Deploy and monitor
- If provider issues persist: Implement simplified single-agent mode
- Consider hybrid: Simple mode for free, advanced for paid

---

## Relation to P2_7B_MODEL_GARBAGE_OUTPUT

This P1 bug **supersedes** the P2 bug. The P2 doc incorrectly blamed the model capacity. The real issues are:

1. **Provider routing** (Together.ai Turbo, not native HF)
2. **Tool execution failure** (possible code bug)
3. **Model hallucination** (consequence of #2, not root cause)

The P2 symptoms are downstream effects of this P1 root cause.

---

## Investigation Timeline

| Time | Finding |
|------|---------|
| 16:00 | Started deep investigation per user request |
| 16:10 | Found Qwen chat template uses XML-style tool_call |
| 16:20 | Confirmed HF API parses tool calls correctly |
| 16:30 | Discovered model routed to Together.ai, not native HF |
| 16:35 | Found premature marker in HuggingFaceChatClient |
| 16:40 | Verified ChatAgent makes tool requests but doesn't execute |
| 16:45 | Documented complete root cause chain |

---

## References

- [HuggingFace Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Together.ai Function Calling](https://docs.together.ai/docs/function-calling)
- [Qwen Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [TGI Tool Calling Issue #2375](https://github.com/huggingface/text-generation-inference/issues/2375)
