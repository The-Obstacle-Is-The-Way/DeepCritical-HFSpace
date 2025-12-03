# P2 Bug: 7B Model Produces Garbage Streaming Output

**Date**: 2025-12-02
**Status**: OPEN - Investigating
**Severity**: P2 (Major - Degrades User Experience)
**Component**: Free Tier / HuggingFace + Multi-Agent Orchestration

---

## Symptoms

When running a research query on Free Tier (Qwen2.5-7B-Instruct), the streaming output shows **garbage tokens** instead of coherent agent reasoning:

```
📡 **STREAMING**: yarg
📡 **STREAMING**: PostalCodes
📡 **STREAMING**: PostalCodes
📡 **STREAMING**: FunctionFlags
📡 **STREAMING**: search_pubmed
📡 **STREAMING**: search_clinical_trials
📡 **STREAMING**: system
📡 **STREAMING**: Transferred to searcher, adopt the persona immediately.
```

The model outputs random tokens like "yarg", "PostalCodes", "FunctionFlags" instead of actual research reasoning.

---

## Reproduction Steps

1. Go to HuggingFace Spaces: https://huggingface.co/spaces/vcms/deepboner
2. Leave API key empty (Free Tier)
3. Click any example query or type a question
4. Click submit
5. Observe streaming output - garbage tokens appear

**Expected**: Coherent agent reasoning like "Searching PubMed for female libido treatments..."
**Actual**: Random tokens like "yarg", "PostalCodes"

---

## Root Cause Analysis

### Primary Cause: 7B Model Too Small for Multi-Agent Prompts

The Qwen2.5-7B-Instruct model has **insufficient reasoning capacity** for the complex multi-agent framework. The system requires the model to:

1. **Adopt agent personas** with specialized instructions
2. **Follow structured workflows** (Search → Judge → Hypothesis → Report)
3. **Make tool calls** (search_pubmed, search_clinical_trials, etc.)
4. **Generate JSON-formatted progress ledgers** for workflow control
5. **Understand manager instructions** and delegate appropriately

A 7B parameter model simply does not have the reasoning depth to handle this. Larger models (70B+) were originally intended, but those are routed to unreliable third-party providers (see `HF_FREE_TIER_ANALYSIS.md`).

### Technical Flow (Where Garbage Appears)

```
User Query
    ↓
AdvancedOrchestrator.run() [advanced.py:247]
    ↓
workflow.run_stream(task) [builds Magentic workflow]
    ↓
MagenticAgentDeltaEvent emitted with event.text
    ↓
Yields AgentEvent(type="streaming", message=event.text) [advanced.py:314-319]
    ↓
Gradio displays: "📡 **STREAMING**: {garbage}"
```

The garbage tokens are **raw model output**. The 7B model is:
- Not following the system prompt
- Outputting partial/incomplete token sequences
- Possibly attempting tool calls but formatting incorrectly
- Hallucinating random words

### Evidence from Microsoft Reference Framework

The Microsoft Agent Framework's `_magentic.py` (lines 1717-1741) shows how agent invocation works:

```python
async for update in agent.run_stream(messages=self._chat_history):
    updates.append(update)
    await self._emit_agent_delta_event(ctx, update)
```

The framework passes through whatever the underlying chat client produces. If the model produces garbage, the framework streams it directly.

### Why Click Example vs Submit Shows Different Initial State

Both code paths go through the same `research_agent()` function in `app.py`. The difference:

- **Example click**: Immediately submits query, so you see garbage quickly
- **Submit button click**: Shows "Starting research (Advanced mode)" banner first, then garbage

Both ultimately produce the same garbage output from the 7B model.

---

## Impact Assessment

| Aspect | Impact |
|--------|--------|
| Free Tier Users | Cannot get usable research results |
| Demo Quality | Appears broken/unprofessional |
| Trust | Users may think the entire system is broken |
| Differentiation | Undermines "free tier works!" messaging |

---

## Potential Solutions

### Option 1: Switch to Better Small Model (Recommended - Quick Fix)

Find a small model that better handles complex instructions. Candidates:

| Model | Size | Tool Calling | Instruction Following |
|-------|------|--------------|----------------------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Yes | Better |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Limited | Good |
| `google/gemma-2-9b-it` | 9B | Yes | Good |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Yes | Better |

**Risk**: 14B model might still be routed to third-party providers. Need to test each.

### Option 2: Simplify Free Tier Architecture

Create a **simpler single-agent mode** for Free Tier:
- Remove multi-agent coordination (Manager, multiple ChatAgents)
- Use a single direct query → search → synthesize flow
- Reduce prompt complexity significantly

**Pros**: More reliable with smaller models
**Cons**: Loses sophisticated multi-agent research capability

### Option 3: Output Filtering/Validation

Add validation layer to detect and filter garbage output:

```python
def is_valid_streaming_token(text: str) -> bool:
    """Check if streaming token appears valid."""
    # Garbage patterns we've seen
    garbage_patterns = ["yarg", "PostalCodes", "FunctionFlags"]
    if any(g in text for g in garbage_patterns):
        return False
    # Check for minimum coherence (has spaces, reasonable length)
    return len(text) > 0 and text.strip()
```

**Pros**: Band-aid fix, quick to implement
**Cons**: Doesn't fix root cause, will miss new garbage patterns

### Option 4: Graceful Degradation

Detect when model output is incoherent and fall back to:
- Returning an error message
- Suggesting user provide an API key
- Using a cached/templated response

### Option 5: Prompt Engineering for 7B Models

Significantly simplify the agent prompts for 7B compatibility:
- Shorter system prompts
- More explicit step-by-step instructions
- Remove abstract concepts
- Use few-shot examples

---

## Recommended Action Plan

### Phase 1: Quick Fix (P2)
1. Test `mistralai/Mistral-7B-Instruct-v0.3` or `Qwen/Qwen2.5-14B-Instruct`
2. Verify they stay on HuggingFace native infrastructure (no third-party routing)
3. Evaluate output quality on sample queries

### Phase 2: Architecture Review (P3)
1. Consider simplified single-agent mode for Free Tier
2. Design graceful degradation when model output is invalid
3. Add output validation layer

### Phase 3: Long-term (P4)
1. Consider hybrid approach: simple mode for free tier, advanced for paid
2. Explore fine-tuning a small model specifically for research agent tasks

---

## Files Involved

| File | Relevance |
|------|-----------|
| `src/orchestrators/advanced.py` | Main orchestrator, streaming event handling |
| `src/clients/huggingface.py` | HuggingFace chat client adapter |
| `src/agents/magentic_agents.py` | Agent definitions and prompts |
| `src/app.py` | Gradio UI, event display |
| `src/utils/config.py` | Model configuration |

---

## Relation to Previous Bugs

- **P0 Repr Bug (RESOLVED)**: Fixed in PR #117 - Was about `<generator object>` appearing due to async generator mishandling
- **P1 HuggingFace Novita Error (RESOLVED)**: Fixed in PR #118 - Was about 72B models being routed to failing third-party providers

This P2 bug is **downstream** of the P1 fix - we fixed the 500 errors by switching to 7B, but now the 7B model doesn't produce quality output.

---

## Questions to Investigate

1. What models in the 7-20B range stay on HuggingFace native infrastructure?
2. Can we detect third-party routing before making the full request?
3. Is the chat template correct for Qwen2.5-7B? (Some models need specific formatting)
4. Are there HuggingFace serverless models specifically optimized for tool calling?

---

## References

- `HF_FREE_TIER_ANALYSIS.md` - Analysis of HuggingFace provider routing
- `CLAUDE.md` - Critical HuggingFace Free Tier section
- Microsoft Agent Framework `_magentic.py` - Reference implementation
