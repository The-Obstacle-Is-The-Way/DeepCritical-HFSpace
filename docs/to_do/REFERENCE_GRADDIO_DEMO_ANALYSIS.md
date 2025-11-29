# Reference: GradioDemo Analysis

> Analysis of code from https://github.com/DeepBoner/GradioDemo
> Purpose: Extract good ideas, understand patterns, avoid mistakes

## Overview

| Metric | Value |
|--------|-------|
| Total lines added | ~7,000 |
| New Python files | +20 |
| Test pass rate | 80% (62 errors due to missing mocks) |
| Integration status | **NOT WIRED IN** |

## Component Catalog

### REDUNDANT (Already have equivalent)

| Component | Lines | What We Have Instead |
|-----------|-------|---------------------|
| `orchestrator/graph_orchestrator.py` | 974 | MagenticBuilder |
| `middleware/budget_tracker.py` | 391 | MagenticBuilder max_round_count |
| `middleware/state_machine.py` | 130 | agents/state.py with contextvars |
| `middleware/workflow_manager.py` | 300 | asyncio.gather() |
| `orchestrator/research_flow.py` (IterativeResearchFlow) | 500 | MagenticOrchestrator |
| HuggingFace integration | various | HFInferenceJudgeHandler |

### POTENTIALLY USEFUL (Ideas to cherry-pick)

#### 1. InputParser (`agents/input_parser.py` - 179 lines)

**Idea**: Detect research mode from query text.

```python
# Key logic (simplified)
research_mode: Literal["iterative", "deep"] = "iterative"
if any(keyword in query.lower() for keyword in [
    "comprehensive", "report", "sections", "analyze", "analysis", "overview", "market"
]):
    research_mode = "deep"
```

**Good pattern**: Heuristic fallback when LLM fails.
**Our implementation**: See Phase 1 in DEEP_RESEARCH_ROADMAP.md

#### 2. PlannerAgent (`orchestrator/planner_agent.py` - 184 lines)

**Idea**: LLM creates section outline for report.

```python
class ReportPlan(BaseModel):
    title: str
    sections: list[ReportSection]
    estimated_time_minutes: int

class ReportSection(BaseModel):
    title: str
    query: str
    description: str
    priority: int
```

**Good pattern**: Structured output with Pydantic models.
**Our implementation**: See Phase 2 in DEEP_RESEARCH_ROADMAP.md

#### 3. DeepResearchFlow (`orchestrator/research_flow.py` - 500 lines)

**Idea**: Run parallel research loops per section.

```python
# Their pattern (simplified)
async def run_parallel_loops(sections: list[ReportSection]):
    tasks = [run_single_loop(s) for s in sections]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Problem**: They built new IterativeResearchFlow instead of reusing MagenticOrchestrator.
**Our implementation**: Just run multiple MagenticOrchestrator instances.

#### 4. LlamaIndex RAG (`services/llamaindex_rag.py` - 454 lines)

**Idea**: Semantic search over collected evidence.

```python
# Their approach
class LlamaIndexRAGService:
    def __init__(self):
        # ChromaDB + LlamaIndex + HuggingFace embeddings
        self.vector_store = ChromaVectorStore(...)
        self.index = VectorStoreIndex(...)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)
```

**Good**: Full-featured RAG with multiple embedding providers.
**Simpler alternative**: Direct ChromaDB + sentence-transformers (no LlamaIndex).
**Our implementation**: See Phase 4 in DEEP_RESEARCH_ROADMAP.md

#### 5. LongWriterAgent (`agents/long_writer.py` - ~300 lines)

**Idea**: Write reports section-by-section to handle length.

```python
class SectionOutput(BaseModel):
    section_content: str
    references: list[str]
    next_section_context: str  # What to avoid repeating

async def write_next_section(
    section_title: str,
    findings: str,
    previous_sections: str,  # Avoid repetition
) -> SectionOutput:
```

**Good pattern**: Passing context to avoid repetition.
**Our implementation**: See Phase 5 in DEEP_RESEARCH_ROADMAP.md

#### 6. ProofreaderAgent (`agents/proofreader.py` - ~200 lines)

**Idea**: Final cleanup pass on report.

```python
# Tasks:
# 1. Remove duplicate information
# 2. Fix citation numbering
# 3. Add executive summary
# 4. Ensure consistent formatting
```

**Good pattern**: Separate concerns - writer writes, proofreader polishes.
**Our implementation**: Optional Phase 6 if needed.

### Graph Architecture (Educational Reference)

The graph system is well-designed in theory:

```python
# Node types
class AgentNode(GraphNode):
    agent: Any  # Pydantic AI agent
    input_transformer: Callable  # Transform input
    output_transformer: Callable  # Transform output

class DecisionNode(GraphNode):
    decision_function: Callable[[Any], str]  # Returns next node ID
    options: list[str]

class ParallelNode(GraphNode):
    parallel_nodes: list[str]  # Run these in parallel
    aggregator: Callable  # Combine results

# Graph structure
class ResearchGraph:
    nodes: dict[str, GraphNode]
    edges: dict[str, list[GraphEdge]]
    entry_node: str
    exit_nodes: list[str]
```

**Why we don't need it**: MagenticBuilder already provides:
- Agent coordination via manager
- Conditional routing (manager decides)
- Multiple participants

This is essentially reimplementing what `agent-framework` already does.

## Key Lessons

### What Went Wrong

1. **Parallel architecture** - Built new system instead of extending existing
2. **Horizontal sprawl** - All infrastructure, nothing wired in
3. **Test mocking** - Tests don't mock API clients properly
4. **No manual testing** - Code never ran end-to-end

### What To Learn From

1. **Pydantic models for structured output** - Good pattern
2. **Heuristic fallbacks** - When LLM fails, have a fallback
3. **Section-by-section writing** - For long reports
4. **RAG for evidence retrieval** - Useful for large evidence sets

### The 7,000 Line vs 500 Line Comparison

**Their approach**:
- New GraphOrchestrator (974 lines)
- New ResearchFlow (999 lines)
- New BudgetTracker (391 lines)
- New StateMachine (130 lines)
- New WorkflowManager (300 lines)
- New agents (InputParser, Writer, LongWriter, Proofreader, etc.)
- Total: ~7,000 lines, not integrated

**Our approach**:
- InputParser (50-100 lines) - extends existing
- PlannerAgent (80-120 lines) - uses ChatAgent pattern
- DeepOrchestrator (100-150 lines) - wraps MagenticOrchestrator
- RAGService (100-150 lines) - simple ChromaDB
- LongWriter (80-100 lines) - extends ReportAgent
- Total: ~500 lines, each phase ships working

## File Locations (for reference)

```
reference_repos/GradioDemo/src/
├── orchestrator/
│   ├── graph_orchestrator.py    # 974 lines - graph execution
│   ├── research_flow.py         # 999 lines - iterative/deep flows
│   └── planner_agent.py         # 184 lines - section planning
├── agents/
│   ├── input_parser.py          # 179 lines - query analysis
│   ├── writer.py                # 210 lines - report writing
│   ├── long_writer.py           # ~300 lines - section writing
│   ├── proofreader.py           # ~200 lines - cleanup
│   └── knowledge_gap.py         # gap detection
├── middleware/
│   ├── budget_tracker.py        # 391 lines - token/time tracking
│   ├── state_machine.py         # 130 lines - workflow state
│   └── workflow_manager.py      # 300 lines - parallel loop mgmt
├── services/
│   └── llamaindex_rag.py        # 454 lines - RAG service
├── tools/
│   └── rag_tool.py              # 191 lines - RAG as search tool
└── agent_factory/
    └── graph_builder.py         # ~400 lines - graph construction
```
