# Implementation Phases: Dual-Mode Agent System

**Date:** November 27, 2025
**Status:** IMPLEMENTATION PLAN (SENIOR AGENT APPROVED)
**Starting Point:** `origin/dev` branch (has Agent Framework code intact)

> **Senior Agent Verdict:** GO - PROCEED WITH OPTION B (CHERRY-PICK)
>
> See `01_ARCHITECTURE_SPEC.md` Section 10 for risks and mitigations.

---

## Phase 0: Branch Setup and Cleanup

**Goal:** Clean slate with correct starting point

### Step 0.1: Close PR #41 (Do Not Merge)

```bash
gh pr close 41 --comment "Closing: Architecture decision changed. Will cherry-pick improvements instead of full refactor."
```

### Step 0.2: Reset Local Environment

```bash
# Ensure we're not on a branch we're about to delete
git checkout main

# Delete the problematic local branches
git branch -D refactor/pydantic-unification
git branch -D feat/pubmed-fulltext

# Reset local dev to match origin/dev (the safe version)
git branch -D dev
git checkout -b dev origin/dev

# Verify we have agent framework code
ls src/agents/  # Should show all agent files
ls src/orchestrator_magentic.py  # Should exist
```

### Step 0.3: Create Fresh Feature Branch

```bash
git checkout -b feat/dual-mode-architecture origin/dev
```

### Verification Checklist

- [ ] PR #41 is closed (not merged)
- [ ] Local `dev` matches `origin/dev`
- [ ] `src/agents/` directory exists with all files
- [ ] `src/orchestrator_magentic.py` exists
- [ ] New branch `feat/dual-mode-architecture` created

---

## Phase 1: Port pydantic-ai Improvements (Cherry-Pick)

**Goal:** Add HuggingFace free tier support without breaking agent framework

### Step 1.1: Update `src/agent_factory/judges.py`

Add the unified `get_model()` function that supports HuggingFace:

```python
# Add to imports
from pydantic_ai.models.huggingface import HuggingFaceModel

# Add/modify get_model() function
def get_model() -> Any:
    """Get LLM model based on configuration. Supports OpenAI, Anthropic, HuggingFace."""
    llm_provider = settings.llm_provider

    if llm_provider == "anthropic":
        provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model, provider=provider)

    if llm_provider == "huggingface":
        # Free tier - uses HF_TOKEN from environment if available
        model_name = settings.huggingface_model or "meta-llama/Llama-3.1-70B-Instruct"
        return HuggingFaceModel(model_name)

    # Default to OpenAI
    if llm_provider != "openai":
        logger.warning("Unknown LLM provider, defaulting to OpenAI", provider=llm_provider)

    openai_provider = OpenAIProvider(api_key=settings.openai_api_key)
    return OpenAIModel(settings.openai_model, provider=openai_provider)
```

### Step 1.2: Update `src/utils/config.py`

Add HuggingFace configuration:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # HuggingFace (free tier)
    huggingface_model: str | None = None
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    @property
    def has_huggingface_key(self) -> bool:
        return bool(self.hf_token)

    @property
    def has_any_llm_key(self) -> bool:
        return self.has_openai_key or self.has_anthropic_key or self.has_huggingface_key
```

### Step 1.3: Add Free Tier Demo

Create `examples/free_tier_demo.py`:

```python
"""Demo of free tier (HuggingFace Inference) capability."""
import asyncio
import os
from pydantic_ai.models.huggingface import HuggingFaceModel
from src.agent_factory.judges import JudgeHandler
from src.orchestrator_factory import create_orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler

async def main():
    print("Starting Free Tier Demo (No OpenAI Key Required)")

    model = HuggingFaceModel("meta-llama/Llama-3.1-8B-Instruct")
    judge_handler = JudgeHandler(model=model)
    search_handler = SearchHandler(tools=[PubMedTool()], timeout=10)

    orchestrator = create_orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=OrchestratorConfig(max_iterations=2),
    )

    async for event in orchestrator.run("What is metformin's mechanism in diabetes?"):
        print(f"[{event.type}] {event.message}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Verification Checklist

- [ ] `judges.py` has `HuggingFaceModel` import and support
- [ ] `config.py` has HuggingFace settings
- [ ] `examples/free_tier_demo.py` created
- [ ] `make check` passes
- [ ] Agent framework code still intact

---

## Phase 2: Update Orchestrator Factory for Mode Selection

**Goal:** Auto-detect mode based on available credentials

### Step 2.1: Update `src/orchestrator_factory.py`

```python
"""Factory for creating orchestrators with mode auto-detection."""
from src.orchestrator import Orchestrator, JudgeHandlerProtocol, SearchHandlerProtocol
from src.utils.config import settings
from src.utils.models import OrchestratorConfig
from src.utils.exceptions import ConfigurationError


def create_orchestrator(
    search_handler: SearchHandlerProtocol,
    judge_handler: JudgeHandlerProtocol,
    config: OrchestratorConfig | None = None,
    mode: str | None = None,
) -> Orchestrator:
    """
    Create orchestrator with auto-mode detection.

    Args:
        search_handler: Search handler
        judge_handler: Judge handler
        config: Optional config
        mode: "simple", "advanced", or None (auto-detect)

    Returns:
        Orchestrator instance (simple or magentic based on mode)
    """
    effective_mode = _determine_mode(mode)

    if effective_mode == "advanced":
        return _create_magentic_orchestrator(search_handler, judge_handler, config)

    return _create_simple_orchestrator(search_handler, judge_handler, config)


def _determine_mode(explicit_mode: str | None) -> str:
    """Determine which mode to use."""
    if explicit_mode:
        return explicit_mode

    # Auto-detect: advanced if paid API key available
    if settings.has_openai_key or settings.has_anthropic_key:
        return "advanced"

    return "simple"


def _create_simple_orchestrator(
    search_handler: SearchHandlerProtocol,
    judge_handler: JudgeHandlerProtocol,
    config: OrchestratorConfig | None,
) -> Orchestrator:
    """Create simple pydantic-ai orchestrator."""
    from src.orchestrator import Orchestrator
    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )


def _create_magentic_orchestrator(
    search_handler: SearchHandlerProtocol,
    judge_handler: JudgeHandlerProtocol,
    config: OrchestratorConfig | None,
) -> Orchestrator:
    """Create advanced multi-agent orchestrator."""
    from src.orchestrator_magentic import MagenticOrchestrator
    return MagenticOrchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )
```

### Verification Checklist

- [ ] Factory auto-detects mode
- [ ] Simple mode works without API keys
- [ ] Advanced mode works with API keys
- [ ] `make check` passes

---

## Phase 3: Fix Agent Framework Integration

**Goal:** Ensure agent framework code works with latest pydantic-ai

### Step 3.1: Update Agent Imports

In `src/agents/tools.py`, `src/agents/judge_agent.py`, etc., ensure imports work:

```python
# These files use agent_framework
# Verify the import works:
from agent_framework import (
    BaseAgent,
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    Role,
    ai_function,
)
```

### Step 3.2: Ensure Agent Framework is in Dependencies

In `pyproject.toml`, add agent-framework as optional dependency:

```toml
[project.optional-dependencies]
# ... existing ...
orchestration = [
    "agent-framework-core>=1.0.0b",
]
```

### Step 3.3: Update MagenticOrchestrator

Ensure `src/orchestrator_magentic.py` uses pydantic-ai internally for structured outputs.

### Step 3.4: (OPTIONAL) Rename "Magentic" to "Advanced"

> **Senior Agent Recommendation:** Rename files to eliminate confusion with the `magentic` PyPI package.

```bash
# Rename orchestrator
git mv src/orchestrator_magentic.py src/orchestrator_advanced.py

# Rename agents file
git mv src/agents/magentic_agents.py src/agents/advanced_agents.py

# Update all imports in:
# - src/orchestrator_factory.py
# - src/app.py
# - tests/
```

**Note:** This is optional for the hackathon. Can be done in a follow-up PR.

### Verification Checklist

- [ ] Agent framework imports work
- [ ] `agent-framework-core` is installable
- [ ] MagenticOrchestrator runs with API key
- [ ] Agents produce structured outputs via pydantic-ai
- [ ] (Optional) Files renamed to avoid "magentic" confusion

---

## Phase 4: Update Gradio UI for Mode Display

**Goal:** Show user which mode is active

### Step 4.1: Update `src/app.py`

```python
def get_mode_indicator() -> str:
    """Return indicator of current operating mode."""
    if settings.has_openai_key or settings.has_anthropic_key:
        return "Advanced Mode (Multi-Agent Orchestration)"
    return "Simple Mode (Free Tier - HuggingFace)"

# In create_app():
with gr.Blocks() as demo:
    gr.Markdown(f"## DeepCritical Research Agent")
    gr.Markdown(f"**Mode:** {get_mode_indicator()}")
    # ... rest of UI
```

### Verification Checklist

- [ ] UI shows current mode
- [ ] Mode changes based on API keys
- [ ] Demo mode works

---

## Phase 5: Testing and Validation

**Goal:** Ensure both modes work correctly

### Step 5.1: Unit Tests

```bash
# Run all tests
make test

# Test simple mode specifically
LLM_PROVIDER=huggingface uv run pytest tests/unit/ -v

# Test advanced mode (requires API key in env)
LLM_PROVIDER=openai uv run pytest tests/unit/ -v
```

### Step 5.2: Integration Tests

```bash
# Test free tier demo
uv run python examples/free_tier_demo.py

# Test full orchestration (requires API key)
uv run python examples/orchestrator_demo/run_agent.py
```

### Step 5.3: Full Quality Check

```bash
make check  # lint + typecheck + test
```

### Verification Checklist

- [ ] All unit tests pass
- [ ] Free tier demo works
- [ ] Advanced demo works (with API key)
- [ ] `make check` passes
- [ ] No regressions in existing functionality

---

## Phase 6: Documentation and Merge

**Goal:** Document and merge to dev

### Step 6.1: Update README

Add section explaining dual-mode operation.

### Step 6.2: Create PR

```bash
git add -A
git commit -m "feat: implement dual-mode architecture (pydantic-ai + agent framework)"
git push origin feat/dual-mode-architecture
gh pr create --base dev --title "feat: Dual-mode architecture" --body "..."
```

### Step 6.3: Review and Merge

- Get CodeRabbit review
- Address feedback
- Merge to dev
- Sync to HuggingFace Spaces

---

## Summary: File Changes by Phase

| Phase | Files Modified/Created |
|-------|------------------------|
| 0 | Branch setup only |
| 1 | `judges.py`, `config.py`, `examples/free_tier_demo.py` |
| 2 | `orchestrator_factory.py` |
| 3 | `pyproject.toml`, agent files, (optional: rename magentic→advanced) |
| 4 | `app.py` |
| 5 | Tests |
| 6 | README, PR |

---

## Risk Mitigation

1. **Before each phase**: Run `make check`
2. **Before committing**: Verify agent framework code intact
3. **If stuck**: Revert to `origin/dev` and start over
4. **Don't force push** to any shared branch
