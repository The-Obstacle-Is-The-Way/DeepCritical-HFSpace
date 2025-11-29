# Immediate Actions Checklist

**Date:** November 27, 2025
**Priority:** Execute in order

---

## Before Starting Implementation

### 1. Close PR #41 (CRITICAL)

```bash
gh pr close 41 --comment "Architecture decision changed. Cherry-picking improvements to preserve both pydantic-ai and Agent Framework capabilities."
```

### 2. Verify HuggingFace Spaces is Safe

```bash
# Should show agent framework files exist
git ls-tree --name-only huggingface-upstream/dev -- src/agents/
git ls-tree --name-only huggingface-upstream/dev -- src/orchestrator_magentic.py
```

Expected output: Files should exist (they do as of this writing).

### 3. Clean Local Environment

```bash
# Switch to main first
git checkout main

# Delete problematic branches
git branch -D refactor/pydantic-unification 2>/dev/null || true
git branch -D feat/pubmed-fulltext 2>/dev/null || true

# Reset local dev to origin/dev
git branch -D dev 2>/dev/null || true
git checkout -b dev origin/dev

# Verify agent framework code exists
ls src/agents/
# Expected: __init__.py, analysis_agent.py, hypothesis_agent.py, judge_agent.py,
#           magentic_agents.py, report_agent.py, search_agent.py, state.py, tools.py

ls src/orchestrator_magentic.py
# Expected: file exists
```

### 4. Create Fresh Feature Branch

```bash
git checkout -b feat/dual-mode-architecture origin/dev
```

---

## Decision Points

Before proceeding, confirm:

1. **For hackathon**: Do we need advanced mode, or is simple mode sufficient?
   - Simple mode = faster to implement, works today
   - Advanced mode = better quality, more work

2. **Timeline**: How much time do we have?
   - If < 1 day: Focus on simple mode only
   - If > 1 day: Implement dual-mode

3. **Dependencies**: Is `agent-framework-core` available?
   - Check: `pip index versions agent-framework-core`
   - If not on PyPI, may need to install from GitHub

---

## Quick Start (Simple Mode Only)

If time is limited, implement only simple mode improvements:

```bash
# On feat/dual-mode-architecture branch

# 1. Update judges.py to add HuggingFace support
# 2. Update config.py to add HF settings
# 3. Create free_tier_demo.py
# 4. Run make check
# 5. Create PR to dev
```

This gives you free-tier capability without touching agent framework code.

---

## Quick Start (Full Dual-Mode)

If time permits, implement full dual-mode:

Follow phases 1-6 in `02_IMPLEMENTATION_PHASES.md`

---

## Emergency Rollback

If anything goes wrong:

```bash
# Reset to safe state
git checkout main
git branch -D feat/dual-mode-architecture
git checkout -b feat/dual-mode-architecture origin/dev
```

Origin/dev is the safe fallback - it has agent framework intact.
