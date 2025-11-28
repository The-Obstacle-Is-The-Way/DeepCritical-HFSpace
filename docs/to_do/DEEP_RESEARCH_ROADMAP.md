# Deep Research Roadmap

> How to properly add GPT-Researcher-style deep research to DeepCritical
> using the EXISTING Magentic + Pydantic AI architecture.

## Current State

We already have:

| Feature | Location | Status |
|---------|----------|--------|
| Multi-agent orchestration | `orchestrator_magentic.py` | Working |
| SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent | `agents/magentic_agents.py` | Working |
| HuggingFace free tier | `agent_factory/judges.py` (HFInferenceJudgeHandler) | Working |
| Budget constraints | MagenticBuilder (max_round_count, max_stall_count) | Built-in |
| Simple mode (linear) | `orchestrator.py` | Working |

## What Deep Research Adds

GPT-Researcher style "deep research" means:

1. **Query Analysis** - Detect if query needs simple lookup vs comprehensive report
2. **Section Planning** - Break complex query into 3-7 parallel research sections
3. **Parallel Research** - Run multiple research loops simultaneously
4. **Long-form Writing** - Synthesize sections into cohesive report
5. **RAG** - Semantic search over accumulated evidence

## Implementation Plan (TDD, Vertical Slices)

### Phase 1: Input Parser (Est. 50-100 lines)

**Goal**: Detect research mode from query.

```python
# src/agents/input_parser.py

class ParsedQuery(BaseModel):
    original_query: str
    improved_query: str
    research_mode: Literal["iterative", "deep"]
    key_entities: list[str]

async def parse_query(query: str) -> ParsedQuery:
    """
    Detect if query needs deep research.

    Deep indicators:
    - "comprehensive", "report", "overview", "analysis"
    - Multiple topics/drugs mentioned
    - Requests for sections/structure

    Iterative indicators:
    - Single focused question
    - "what is", "how does", "find"
    """
```

**Test first**:
```python
def test_parse_query_detects_deep_mode():
    result = await parse_query("Write a comprehensive report on Alzheimer's treatments")
    assert result.research_mode == "deep"

def test_parse_query_detects_iterative_mode():
    result = await parse_query("What is the mechanism of metformin?")
    assert result.research_mode == "iterative"
```

**Wire in**:
```python
# In app.py or orchestrator_factory.py
parsed = await parse_query(user_query)
if parsed.research_mode == "deep":
    orchestrator = create_deep_orchestrator()
else:
    orchestrator = create_orchestrator()  # existing
```

---

### Phase 2: Section Planner (Est. 80-120 lines)

**Goal**: Create report outline for deep research.

```python
# src/agents/planner.py

class ReportSection(BaseModel):
    title: str
    query: str  # Search query for this section
    description: str

class ReportPlan(BaseModel):
    title: str
    sections: list[ReportSection]

# Use existing ChatAgent pattern from magentic_agents.py
def create_planner_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    return ChatAgent(
        name="PlannerAgent",
        description="Creates structured report outlines",
        instructions="""Given a research query, create a report plan with 3-7 sections.
        Each section should have:
        - A clear title
        - A focused search query
        - Brief description of what to cover

        Example for "Alzheimer's drug repurposing":
        1. Current Treatment Landscape
        2. Mechanism-Based Candidates (targeting amyloid, tau, inflammation)
        3. Clinical Trial Evidence
        4. Safety Considerations
        5. Emerging Research Directions
        """,
        chat_client=client,
    )
```

**Test first**:
```python
def test_planner_creates_sections():
    plan = await planner.create_plan("Comprehensive Alzheimer's drug repurposing report")
    assert len(plan.sections) >= 3
    assert all(s.query for s in plan.sections)
```

**Wire in**: Used by Phase 3.

---

### Phase 3: Parallel Research Flow (Est. 100-150 lines)

**Goal**: Run multiple MagenticOrchestrator instances in parallel.

```python
# src/orchestrator_deep.py

class DeepResearchOrchestrator:
    """
    Runs parallel research loops using EXISTING MagenticOrchestrator.

    NOT a new orchestration system - just a wrapper that:
    1. Plans sections
    2. Runs existing orchestrator per section (in parallel)
    3. Aggregates results
    """

    def __init__(self, max_parallel: int = 5):
        self.planner = create_planner_agent()
        self.max_parallel = max_parallel

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        # 1. Create plan
        plan = await self.planner.create_plan(query)
        yield AgentEvent(type="planning", message=f"Created {len(plan.sections)} section plan")

        # 2. Run parallel research (reuse existing orchestrator!)
        from src.orchestrator_magentic import MagenticOrchestrator

        async def research_section(section: ReportSection) -> str:
            orchestrator = MagenticOrchestrator(max_rounds=5)  # Fewer rounds per section
            result = ""
            async for event in orchestrator.run(section.query):
                if event.type == "complete":
                    result = event.message
            return result

        # Run in parallel with semaphore
        semaphore = asyncio.Semaphore(self.max_parallel)
        async def bounded_research(section):
            async with semaphore:
                return await research_section(section)

        results = await asyncio.gather(*[
            bounded_research(s) for s in plan.sections
        ])

        # 3. Aggregate
        yield AgentEvent(
            type="complete",
            message=self._aggregate_sections(plan, results)
        )
```

**Key insight**: We're NOT replacing MagenticOrchestrator. We're running multiple instances of it.

**Test first**:
```python
@pytest.mark.integration
async def test_deep_orchestrator_runs_parallel():
    orchestrator = DeepResearchOrchestrator(max_parallel=2)
    events = [e async for e in orchestrator.run("Comprehensive Alzheimer's report")]
    assert any(e.type == "planning" for e in events)
    assert any(e.type == "complete" for e in events)
```

---

### Phase 4: RAG Integration (Est. 100-150 lines)

**Goal**: Semantic search over accumulated evidence.

```python
# src/services/rag.py

class RAGService:
    """
    Simple RAG using ChromaDB + sentence-transformers.
    No LlamaIndex dependency - keep it lightweight.
    """

    def __init__(self):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("evidence")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_evidence(self, evidence: list[Evidence]) -> int:
        """Add evidence to vector store, return count added."""
        # Dedupe by URL
        existing = set(self.collection.get()["ids"])
        new_evidence = [e for e in evidence if e.citation.url not in existing]

        if not new_evidence:
            return 0

        self.collection.add(
            ids=[e.citation.url for e in new_evidence],
            documents=[e.content for e in new_evidence],
            metadatas=[{"title": e.citation.title, "source": e.citation.source} for e in new_evidence],
        )
        return len(new_evidence)

    def search(self, query: str, n_results: int = 5) -> list[Evidence]:
        """Semantic search for relevant evidence."""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        # Convert back to Evidence objects
        ...
```

**Wire in as tool**:
```python
# Add to SearchAgent's tools
def rag_search(query: str, n_results: int = 5) -> str:
    """Search previously collected evidence for relevant information."""
    service = get_rag_service()
    results = service.search(query, n_results)
    return format_evidence(results)

# In magentic_agents.py
ChatAgent(
    tools=[search_pubmed, search_clinical_trials, search_preprints, rag_search],  # ADD RAG
)
```

---

### Phase 5: Long Writer (Est. 80-100 lines)

**Goal**: Write longer reports section-by-section.

```python
# Extend existing ReportAgent or create LongWriterAgent

def create_long_writer_agent() -> ChatAgent:
    return ChatAgent(
        name="LongWriterAgent",
        description="Writes detailed report sections with proper citations",
        instructions="""Write a detailed section for a research report.

        You will receive:
        - Section title
        - Relevant evidence/findings
        - What previous sections covered (to avoid repetition)

        Output:
        - 500-1000 words per section
        - Proper citations [1], [2], etc.
        - Smooth transitions
        - No repetition of earlier content
        """,
        tools=[get_bibliography, rag_search],
    )
```

---

## What NOT To Build

These are REDUNDANT with existing Magentic system:

| Component | Why Skip |
|-----------|----------|
| GraphOrchestrator | MagenticBuilder already handles agent coordination |
| BudgetTracker | MagenticBuilder has max_round_count, max_stall_count |
| WorkflowManager | asyncio.gather() + Semaphore is simpler |
| StateMachine | contextvars already used in agents/state.py |
| New agent primitives | ChatAgent pattern already works |

## Implementation Order

```
Week 1: Phase 1 (InputParser) - Ship it working
Week 2: Phase 2 (Planner) - Ship it working
Week 3: Phase 3 (Parallel Flow) - Ship it working
Week 4: Phase 4 (RAG) - Ship it working
Week 5: Phase 5 (LongWriter) - Ship it working
```

Each phase:
1. Write tests first
2. Implement minimal code
3. Wire into app.py
4. Manual test
5. PR with <200 lines
6. Ship

## References

- GPT-Researcher: https://github.com/assafelovic/gpt-researcher
- LangGraph patterns: https://python.langchain.com/docs/langgraph
- Your existing Magentic setup: `src/orchestrator_magentic.py`

## Why This Approach

1. **Builds on existing working code** - Don't replace, extend
2. **Each phase ships value** - User sees improvement after each PR
3. **Tests prove it works** - Not "trust me it imports"
4. **Minimal new abstractions** - Reuse ChatAgent, MagenticOrchestrator
5. **~500 total lines** vs 7,000 lines of parallel infrastructure
