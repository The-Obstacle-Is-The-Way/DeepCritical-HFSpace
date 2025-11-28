# Senior Agent Review Prompt

Copy and paste everything below this line to a fresh Claude/AI session:

---

## Context

I am a junior developer working on a HuggingFace hackathon project called DeepCritical. We made a significant architectural mistake and are now trying to course-correct. I need you to act as a **senior staff engineer** and critically review our proposed solution.

## The Situation

We almost merged a refactor that would have **deleted** our multi-agent orchestration capability, mistakenly believing that `pydantic-ai` (a library for structured LLM outputs) and Microsoft's `agent-framework` (a framework for multi-agent orchestration) were mutually exclusive alternatives.

**They are not.** They are complementary:
- `pydantic-ai` ensures LLM responses match Pydantic schemas (type-safe outputs)
- `agent-framework` orchestrates multiple agents working together (coordination layer)

We now want to implement a **dual-mode architecture** where:
- **Simple Mode (No API key):** Uses only pydantic-ai with HuggingFace free tier
- **Advanced Mode (With API key):** Uses Microsoft Agent Framework for orchestration, with pydantic-ai inside each agent for structured outputs

## Your Task

Please perform a **deep, critical review** of:

1. **The architecture diagram** (image attached: `assets/magentic-pydantic.png`)
2. **Our documentation** (4 files listed below)
3. **The actual codebase** to verify our claims

## Specific Questions to Answer

### Architecture Validation
1. Is our understanding correct that pydantic-ai and agent-framework are complementary, not competing?
2. Does the dual-mode architecture diagram accurately represent how these should integrate?
3. Are there any architectural flaws or anti-patterns in our proposed design?

### Documentation Accuracy
4. Are the branch states we documented accurate? (Check `git log`, `git ls-tree`)
5. Is our understanding of what code exists where correct?
6. Are the implementation phases realistic and in the correct order?
7. Are there any missing steps or dependencies we overlooked?

### Codebase Reality Check
8. Does `origin/dev` actually have the agent framework code intact? Verify by checking:
   - `git ls-tree origin/dev -- src/agents/`
   - `git ls-tree origin/dev -- src/orchestrator_magentic.py`
9. What does the current `src/agents/` code actually import? Does it use `agent_framework` or `agent-framework-core`?
10. Is the `agent-framework-core` package actually available on PyPI, or do we need to install from source?

### Implementation Feasibility
11. Can the cherry-pick strategy we outlined actually work, or are there merge conflicts we're not seeing?
12. Is the mode auto-detection logic sound?
13. What are the risks we haven't identified?

### Critical Errors Check
14. Did we miss anything critical in our analysis?
15. Are there any factual errors in our documentation?
16. Would a Google/DeepMind senior engineer approve this plan, or would they flag issues?

## Files to Review

Please read these files in order:

1. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/docs/brainstorming/magentic-pydantic/00_SITUATION_AND_PLAN.md`
2. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/docs/brainstorming/magentic-pydantic/01_ARCHITECTURE_SPEC.md`
3. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/docs/brainstorming/magentic-pydantic/02_IMPLEMENTATION_PHASES.md`
4. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/docs/brainstorming/magentic-pydantic/03_IMMEDIATE_ACTIONS.md`

And the architecture diagram:
5. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/assets/magentic-pydantic.png`

## Reference Repositories to Consult

We have local clones of the source-of-truth repositories:

- **Original DeepCritical:** `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/reference_repos/DeepCritical/`
- **Microsoft Agent Framework:** `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/reference_repos/agent-framework/`
- **Microsoft AutoGen:** `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/reference_repos/autogen-microsoft/`

Please cross-reference our hackathon fork against these to verify architectural alignment.

## Codebase to Analyze

Our hackathon fork is at:
`/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/DeepCritical-1/`

Key files to examine:
- `src/agents/` - Agent framework integration
- `src/agent_factory/judges.py` - pydantic-ai integration
- `src/orchestrator.py` - Simple mode orchestrator
- `src/orchestrator_magentic.py` - Advanced mode orchestrator
- `src/orchestrator_factory.py` - Mode selection
- `pyproject.toml` - Dependencies

## Expected Output

Please provide:

1. **Validation Summary:** Is our plan sound? (YES/NO with explanation)
2. **Errors Found:** List any factual errors in our documentation
3. **Missing Items:** What did we overlook?
4. **Risk Assessment:** What could go wrong?
5. **Recommended Changes:** Specific edits to our documentation or plan
6. **Go/No-Go Recommendation:** Should we proceed with this plan?

## Tone

Be brutally honest. If our plan is flawed, say so directly. We would rather know now than after implementation. Don't soften criticism - we need accuracy.

---

END OF PROMPT
