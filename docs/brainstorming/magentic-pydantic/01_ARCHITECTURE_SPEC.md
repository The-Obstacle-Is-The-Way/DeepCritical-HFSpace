# Architecture Specification: Dual-Mode Agent System

**Date:** November 27, 2025
**Status:** SPECIFICATION
**Goal:** Graceful degradation from full multi-agent orchestration to simple single-agent mode

---

## 1. Core Concept: Two Operating Modes

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                                 │
│                            │                                        │
│                            ▼                                        │
│                   ┌─────────────────┐                               │
│                   │  Mode Selection │                               │
│                   │  (Auto-detect)  │                               │
│                   └────────┬────────┘                               │
│                            │                                        │
│            ┌───────────────┴───────────────┐                        │
│            │                               │                        │
│            ▼                               ▼                        │
│   ┌─────────────────┐             ┌─────────────────┐               │
│   │   SIMPLE MODE   │             │  ADVANCED MODE  │               │
│   │  (Free Tier)    │             │  (Paid Tier)    │               │
│   │                 │             │                 │               │
│   │  pydantic-ai    │             │  MS Agent Fwk   │               │
│   │  single-agent   │             │  + pydantic-ai  │               │
│   │  loop           │             │  multi-agent    │               │
│   └─────────────────┘             └─────────────────┘               │
│            │                               │                        │
│            └───────────────┬───────────────┘                        │
│                            ▼                                        │
│                   ┌─────────────────┐                               │
│                   │  Research Report │                              │
│                   │  with Citations  │                              │
│                   └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mode Comparison

| Aspect | Simple Mode | Advanced Mode |
|--------|-------------|---------------|
| **Trigger** | No API key OR `LLM_PROVIDER=huggingface` | OpenAI API key present (currently OpenAI only) |
| **Framework** | pydantic-ai only | Microsoft Agent Framework + pydantic-ai |
| **Architecture** | Single orchestrator loop | Multi-agent coordination |
| **Agents** | One agent does Search→Judge→Report | SearchAgent, JudgeAgent, ReportAgent, AnalysisAgent |
| **State Management** | Simple dict | Thread-safe `MagenticState` with context vars |
| **Quality** | Good (functional) | Better (specialized agents, coordination) |
| **Cost** | Free (HuggingFace Inference) | Paid (OpenAI/Anthropic) |
| **Use Case** | Demos, hackathon, budget-constrained | Production, research quality |

---

## 3. Simple Mode Architecture (pydantic-ai Only)

```text
┌─────────────────────────────────────────────────────┐
│                  Orchestrator                       │
│                                                     │
│   while not sufficient and iteration < max:        │
│       1. SearchHandler.execute(query)              │
│       2. JudgeHandler.assess(evidence)    ◄── pydantic-ai Agent  │
│       3. if sufficient: break                      │
│       4. query = judge.next_queries                │
│                                                     │
│   return ReportGenerator.generate(evidence)        │
└─────────────────────────────────────────────────────┘
```

**Components:**
- `src/orchestrator.py` - Simple loop orchestrator
- `src/agent_factory/judges.py` - JudgeHandler with pydantic-ai
- `src/tools/search_handler.py` - Scatter-gather search
- `src/tools/pubmed.py`, `clinicaltrials.py`, `europepmc.py` - Search tools

---

## 4. Advanced Mode Architecture (MS Agent Framework + pydantic-ai)

```text
┌─────────────────────────────────────────────────────────────────────┐
│              Microsoft Agent Framework Orchestrator                 │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ SearchAgent │───▶│ JudgeAgent  │───▶│ ReportAgent │            │
│   │ (BaseAgent) │    │ (BaseAgent) │    │ (BaseAgent) │            │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│          │                  │                  │                    │
│          ▼                  ▼                  ▼                    │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ pydantic-ai │    │ pydantic-ai │    │ pydantic-ai │            │
│   │ Agent()     │    │ Agent()     │    │ Agent()     │            │
│   │ output_type=│    │ output_type=│    │ output_type=│            │
│   │ SearchResult│    │ JudgeAssess │    │ Report      │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   Shared State: MagenticState (thread-safe via contextvars)        │
│   - evidence: list[Evidence]                                       │
│   - embedding_service: EmbeddingService                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Components:**
- `src/orchestrator_magentic.py` - Multi-agent orchestrator
- `src/agents/search_agent.py` - SearchAgent (BaseAgent)
- `src/agents/judge_agent.py` - JudgeAgent (BaseAgent)
- `src/agents/report_agent.py` - ReportAgent (BaseAgent)
- `src/agents/analysis_agent.py` - AnalysisAgent (BaseAgent)
- `src/agents/state.py` - Thread-safe state management
- `src/agents/tools.py` - @ai_function decorated tools

---

## 5. Mode Selection Logic

```python
# src/orchestrator_factory.py (actual implementation)

def create_orchestrator(
    search_handler: SearchHandlerProtocol | None = None,
    judge_handler: JudgeHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic", "advanced"] | None = None,
) -> Any:
    """
    Auto-select orchestrator based on available credentials.

    Priority:
    1. If mode explicitly set, use that
    2. If OpenAI key available -> Advanced Mode (currently OpenAI only)
    3. Otherwise -> Simple Mode (HuggingFace free tier)
    """
    effective_mode = _determine_mode(mode)

    if effective_mode == "advanced":
        orchestrator_cls = _get_magentic_orchestrator_class()
        return orchestrator_cls(max_rounds=config.max_iterations if config else 10)

    # Simple mode requires handlers
    if search_handler is None or judge_handler is None:
        raise ValueError("Simple mode requires search_handler and judge_handler")

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )
```

---

## 6. Shared Components (Both Modes Use)

These components work in both modes:

| Component | Purpose |
|-----------|---------|
| `src/tools/pubmed.py` | PubMed search |
| `src/tools/clinicaltrials.py` | ClinicalTrials.gov search |
| `src/tools/europepmc.py` | Europe PMC search |
| `src/tools/search_handler.py` | Scatter-gather orchestration |
| `src/tools/rate_limiter.py` | Rate limiting |
| `src/utils/models.py` | Evidence, Citation, JudgeAssessment |
| `src/utils/config.py` | Settings |
| `src/services/embeddings.py` | Vector search (optional) |

---

## 7. pydantic-ai Integration Points

Both modes use pydantic-ai for structured LLM outputs:

```python
# In JudgeHandler (both modes)
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

class JudgeHandler:
    def __init__(self, model: Any = None):
        self.model = model or get_model()  # Auto-selects based on config
        self.agent = Agent(
            model=self.model,
            output_type=JudgeAssessment,  # Structured output!
            system_prompt=SYSTEM_PROMPT,
        )

    async def assess(self, question: str, evidence: list[Evidence]) -> JudgeAssessment:
        result = await self.agent.run(format_prompt(question, evidence))
        return result.output  # Guaranteed to be JudgeAssessment
```

---

## 8. Microsoft Agent Framework Integration Points

Advanced mode wraps pydantic-ai agents in BaseAgent:

```python
# In JudgeAgent (advanced mode only)
from agent_framework import BaseAgent, AgentRunResponse, ChatMessage, Role

class JudgeAgent(BaseAgent):
    def __init__(self, judge_handler: JudgeHandlerProtocol):
        super().__init__(
            name="JudgeAgent",
            description="Evaluates evidence quality",
        )
        self._handler = judge_handler  # Uses pydantic-ai internally

    async def run(self, messages, **kwargs) -> AgentRunResponse:
        question = extract_question(messages)
        evidence = self._evidence_store.get("current", [])

        # Delegate to pydantic-ai powered handler
        assessment = await self._handler.assess(question, evidence)

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=format_response(assessment))],
            additional_properties={"assessment": assessment.model_dump()},
        )
```

---

## 9. Benefits of This Architecture

1. **Graceful Degradation**: Works without API keys (free tier)
2. **Progressive Enhancement**: Better with API keys (orchestration)
3. **Code Reuse**: pydantic-ai handlers shared between modes
4. **Hackathon Ready**: Demo works without requiring paid keys
5. **Production Ready**: Full orchestration available when needed
6. **Future Proof**: Can add more agents to advanced mode
7. **Testable**: Simple mode is easier to unit test

---

## 10. Known Risks and Mitigations

> **From Senior Agent Review**

### 10.1 Bridge Complexity (MEDIUM)

**Risk:** In Advanced Mode, agents (Agent Framework) wrap handlers (pydantic-ai). Both are async. Context variables (`MagenticState`) must propagate correctly through the pydantic-ai call stack.

**Mitigation:**
- pydantic-ai uses standard Python `contextvars`, which naturally propagate through `await` chains
- Test context propagation explicitly in integration tests
- If issues arise, pass state explicitly rather than via context vars

### 10.2 Integration Drift (MEDIUM)

**Risk:** Simple Mode and Advanced Mode might diverge in behavior over time (e.g., Simple Mode uses logic A, Advanced Mode uses logic B).

**Mitigation:**
- Both modes MUST call the exact same underlying Tools (`src/tools/*`) and Handlers (`src/agent_factory/*`)
- Handlers are the single source of truth for business logic
- Agents are thin wrappers that delegate to handlers

### 10.3 Testing Burden (LOW-MEDIUM)

**Risk:** Two distinct orchestrators (`src/orchestrator.py` and `src/orchestrator_magentic.py`) doubles integration testing surface area.

**Mitigation:**
- Unit test handlers independently (shared code)
- Integration tests for each mode separately
- End-to-end tests verify same output for same input (determinism permitting)

### 10.4 Dependency Conflicts (LOW)

**Risk:** `agent-framework-core` might conflict with `pydantic-ai`'s dependencies (e.g., different pydantic versions).

**Status:** Both use `pydantic>=2.x`. Should be compatible.

---

## 11. Naming Clarification

> See `00_SITUATION_AND_PLAN.md` Section 4 for full details.

**Important:** The codebase uses "magentic" in file names (`orchestrator_magentic.py`, `magentic_agents.py`) but this refers to our internal naming for Microsoft Agent Framework integration, **NOT** the `magentic` PyPI package.

**Future action:** Rename to `orchestrator_advanced.py` to eliminate confusion.
