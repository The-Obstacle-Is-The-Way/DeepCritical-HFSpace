# P0 BUG: Simple Mode Synthesis Bypass (WILL BE FIXED BY UNIFIED ARCHITECTURE)

**Status**: BLOCKED - Waiting for upstream PR #2566
**Priority**: P0 (Demo-blocking)
**Discovered**: 2025-12-01
**GitHub Issue**: [#113](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/113)

---

## Current State

**`simple.py` is DELETED.** This bug existed in the old Simple Mode code.

The bug will NOT be fixed by restoring Simple Mode. Instead, it will be **automatically fixed** when we complete the unified architecture (after upstream PR #2566 merges).

---

## The Bug (Historical)

When HuggingFace Inference API failed, Simple Mode's `_should_synthesize()` ignored forced synthesis signals due to overly strict thresholds.

```text
✅ JUDGE_COMPLETE: Assessment: synthesize (confidence: 10%)
🔄 LOOPING: Gathering more evidence...  ← BUG: Should have synthesized!
```

---

## Why Unified Architecture Fixes This

| Architecture | How Termination Works |
|--------------|----------------------|
| **Old (Simple Mode)** | Custom `_should_synthesize()` with buggy thresholds |
| **New (Unified)** | Manager agent respects "SUFFICIENT EVIDENCE" signals |

The Manager agent in Advanced Mode already works correctly. By completing the unified architecture with HuggingFace support, we inherit that correct behavior.

**No need to patch `_should_synthesize()` because the code is deleted.**

---

## Path Forward

1. **Wait** for upstream PR #2566 to merge (fixes repr bug)
2. **Update** `agent-framework` dependency
3. **Verify** Advanced Mode + HuggingFace works
4. **Done** - This bug is gone (no `_should_synthesize()` thresholds)

---

## Related

| Reference | Description |
|-----------|-------------|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Current state and unified plan |
| [SPEC_16](../specs/SPEC_16_UNIFIED_CHAT_CLIENT_ARCHITECTURE.md) | Unified architecture spec |
| [Issue #105](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/issues/105) | GitHub tracking |
| [Upstream #2562](https://github.com/microsoft/agent-framework/issues/2562) | Framework bug |
| [Upstream PR #2566](https://github.com/microsoft/agent-framework/pull/2566) | Framework fix |
