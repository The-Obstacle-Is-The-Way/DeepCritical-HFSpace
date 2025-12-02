# P1 Bug: Simple Mode Removal Breaks Free Tier UX

**Severity**: P1 (High) - Free Tier shows garbage repr output instead of clean Simple mode
**Status**: OPEN - Requires design decision
**Discovered**: 2025-12-01
**Investigator**: Claude Code

## Executive Summary

The Unified Architecture (SPEC-16) deprecated Simple Mode and routes ALL users to Advanced Mode. When no API key is provided, Advanced Mode falls back to HuggingFace Free Tier, which triggers the upstream agent-framework repr bug (#2562). This results in degraded UX compared to the old Simple Mode.

## Observed Behavior

### MCP-1st-Birthday Space (OLD - Hackathon)
```
📚 **SEARCH_COMPLETE**: Found 21 new sources (21 total)
🧠 **JUDGING**: Evaluating evidence (Memory: 21 docs)...
✅ **JUDGE_COMPLETE**: Assessment: continue (confidence: 80%)
```
- Clean, readable output
- Uses Simple Mode (auto-detected when no API key)

### VibecoderMcSwaggins Space (NEW - Current)
```
📚 **SEARCH_COMPLETE**: searcher: <agent_framework._types.ChatMessage object at 0x7fd3f8617b10>
📚 **SEARCH_COMPLETE**: searcher: <agent_framework._types.ChatMessage object at 0x7fd3f8617dd0>
```
- Garbage repr output
- Uses Advanced Mode (forced by SPEC-16)
- Falls back to HuggingFace which triggers upstream bug

## Root Cause Analysis

### 1. SPEC-16 Unified Architecture (Intentional Change)

**File**: `src/orchestrators/factory.py`

```python
# OLD (MCP Space - has auto-detection):
def _determine_mode(explicit_mode: str | None, api_key: str | None) -> str:
    # Auto-detect: advanced if paid API key available
    if settings.has_openai_key or (api_key and api_key.startswith("sk-")):
        return "advanced"
    return "simple"  # <-- Falls back to Simple Mode!

# NEW (Current - always advanced):
def _determine_mode(explicit_mode: str | None) -> str:
    # "simple" is deprecated -> upgrade to "advanced"
    return "advanced"  # <-- ALWAYS Advanced Mode!
```

### 2. Advanced Mode + HuggingFace = Upstream Bug

When Advanced Mode uses HuggingFace Free Tier:
1. `MagenticAgentExecutor._invoke_agent()` creates events
2. Tool-call-only messages have empty `.text`
3. Framework bug: `text = last.text or str(last)` produces repr
4. Our `_extract_text()` workaround filters repr but can't recover lost data

**Upstream Fix**: PR #2566 (waiting for merge)

## User Confusion: Examples vs Chat Button

The user observed different behavior:
- **Clicking Example** → Seemed to show different output
- **Clicking Chat** → Shows repr garbage with progress timer

### Possible Explanations

1. **Caching**: Gradio may cache example outputs from before the bug appeared
2. **Different Spaces**: User may have been comparing MCP (old) vs Vibecoder (new)
3. **Timing**: HuggingFace Spaces rebuild delay

### Code Path (Same for Both)

Both examples and manual chat call the same function:

```python
# src/app.py:279
demo = gr.ChatInterface(
    fn=research_agent,  # <-- Same function for both
    examples=[
        ["What drugs improve female libido post-menopause?", "sexual_health", None, None],
        ...
    ],
)
```

Examples pass `api_key=None`, so:
1. `configure_orchestrator(user_api_key=None)` is called
2. Falls through to "Free Tier" branch
3. `create_orchestrator(mode="advanced")` is called
4. Advanced Mode with HuggingFace is created
5. Upstream bug produces repr garbage

## Impact Assessment

| Aspect | Old (Simple Mode) | New (Advanced + HF) |
|--------|-------------------|---------------------|
| Output clarity | Clean, readable | Repr garbage |
| Tool execution | Works | Works (bug is display only) |
| Research completion | Works | Works (usually) |
| User experience | Good | Confusing/broken |

## Options for Resolution

### Option A: Wait for Upstream Fix (Current Approach)
- PR #2566 will fix the repr bug
- Once merged and released, update agent-framework
- Advanced Mode will work cleanly

**Pros**: Clean solution, no code changes needed
**Cons**: Dependent on Microsoft's release timeline

### Option B: Restore Simple Mode for Free Tier
- Modify `_determine_mode()` to return "simple" when no API key
- Keep Advanced Mode for paid users

```python
def _determine_mode(explicit_mode: str | None, api_key: str | None) -> str:
    if explicit_mode == "hierarchical":
        return "hierarchical"

    # Restore auto-detection for better Free Tier UX
    if settings.has_openai_key or (api_key and api_key.startswith("sk-")):
        return "advanced"

    return "simple"  # Free tier gets Simple Mode (no repr bug)
```

**Pros**: Immediate fix, better UX for free users
**Cons**: Diverges from SPEC-16, maintains two code paths

### Option C: Strengthen _extract_text() Workaround
- Already implemented but limited (contents lost in framework)
- Can only filter repr, can't show tool names

**Pros**: Already done
**Cons**: Still shows empty messages instead of tool info

## Files Involved

| File | Role |
|------|------|
| `src/orchestrators/factory.py:76-90` | Mode determination (always "advanced") |
| `src/app.py:101-108` | Orchestrator creation |
| `src/orchestrators/advanced.py:336-385` | `_extract_text()` workaround |
| `src/clients/huggingface.py` | HuggingFace client (works correctly) |

## Related Issues

| Issue | Description | Status |
|-------|-------------|--------|
| P0 HF Tool Calling | Our fix for `_convert_messages()` | FIXED |
| Upstream #2562 | Framework repr bug | OPEN |
| Upstream #2566 | Framework fix PR | Pending Merge |

## Verification Steps

1. Visit https://huggingface.co/spaces/VibecoderMcSwaggins/DeepBoner
2. Do NOT enter an API key
3. Click any example or type a question
4. Observe repr garbage in output

## Recommendation

**Short-term**: Wait for upstream #2566 to merge (has 1 approval, all checks passed)

**Medium-term**: If #2566 doesn't merge within 1 week, consider Option B (restore Simple Mode for free tier)

**Long-term**: Once upstream is fixed, the Unified Architecture will work as intended

---

## References

- [SPEC-16: Unified Architecture](../specs/SPEC_16_UNIFIED_ARCHITECTURE.md) (if exists)
- [P0 HuggingFace Tool Calling Bug](./P0_HUGGINGFACE_TOOL_CALLING_BROKEN.md)
- [Upstream Issue #2562](https://github.com/microsoft/agent-framework/issues/2562)
- [Upstream Fix PR #2566](https://github.com/microsoft/agent-framework/pull/2566)
- [MCP Space (old, working)](https://huggingface.co/spaces/MCP-1st-Birthday/DeepBoner)
- [VibecoderMcSwaggins Space (new, broken)](https://huggingface.co/spaces/VibecoderMcSwaggins/DeepBoner)
