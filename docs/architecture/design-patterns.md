# Design Patterns & Technical Decisions
## Explicit Answers to Architecture Questions

---

## Purpose of This Document

This document explicitly answers all the "design pattern" questions raised in team discussions. It provides clear technical decisions with rationale.

---

## 1. Primary Architecture Pattern

### Decision: Orchestrator with Search-Judge Loop

**Pattern Name**: Iterative Research Orchestrator

**Structure**:
```
┌─────────────────────────────────────┐
│    Research Orchestrator            │
│  ┌───────────────────────────────┐  │
│  │  Search Strategy Planner      │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │  Tool Coordinator             │  │
│  │  - PubMed Search              │  │
│  │  - Web Search                 │  │
│  │  - Clinical Trials            │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │  Evidence Aggregator          │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │  Quality Judge                │  │
│  │  (LLM-based assessment)       │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│       Loop or Synthesize?            │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │  Report Generator             │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Why NOT single-agent?**
- Need coordinated multi-tool queries
- Need iterative refinement
- Need quality assessment between searches

**Why NOT pure ReAct?**
- Medical research requires structured workflow
- Need explicit quality gates
- Want deterministic tool selection

**Why THIS pattern?**
- Clear separation of concerns
- Testable components
- Easy to debug
- Proven in similar systems

---

## 2. Tool Selection & Orchestration Pattern

### Decision: Static Tool Registry with Dynamic Selection

**Pattern**:
```python
class ToolRegistry:
    """Central registry of available research tools"""
    tools = {
        'pubmed': PubMedSearchTool(),
        'web': WebSearchTool(),
        'trials': ClinicalTrialsTool(),
        'drugs': DrugInfoTool(),
    }

class Orchestrator:
    def select_tools(self, question: str, iteration: int) -> List[Tool]:
        """Dynamically choose tools based on context"""
        if iteration == 0:
            # First pass: broad search
            return [tools['pubmed'], tools['web']]
        else:
            # Refinement: targeted search
            return self.judge.recommend_tools(question, context)
```

**Why NOT on-the-fly agent factories?**
- 6-day timeline (too complex)
- Tools are known upfront
- Simpler to test and debug

**Why NOT single tool?**
- Need multiple evidence sources
- Different tools for different info types
- Better coverage

**Why THIS pattern?**
- Balance flexibility vs simplicity
- Tools can be added easily
- Selection logic is transparent

---

## 3. Judge Pattern

### Decision: Dual-Judge System (Quality + Budget)

**Pattern**:
```python
class QualityJudge:
    """LLM-based evidence quality assessment"""

    def is_sufficient(self, question: str, evidence: List[Evidence]) -> bool:
        """Main decision: do we have enough?"""
        return (
            self.has_mechanism_explanation(evidence) and
            self.has_drug_candidates(evidence) and
            self.has_clinical_evidence(evidence) and
            self.confidence_score(evidence) > threshold
        )

    def identify_gaps(self, question: str, evidence: List[Evidence]) -> List[str]:
        """What's missing?"""
        gaps = []
        if not self.has_mechanism_explanation(evidence):
            gaps.append("disease mechanism")
        if not self.has_drug_candidates(evidence):
            gaps.append("potential drug candidates")
        if not self.has_clinical_evidence(evidence):
            gaps.append("clinical trial data")
        return gaps

class BudgetJudge:
    """Resource constraint enforcement"""

    def should_stop(self, state: ResearchState) -> bool:
        """Hard limits"""
        return (
            state.tokens_used >= max_tokens or
            state.iterations >= max_iterations or
            state.time_elapsed >= max_time
        )
```

**Why NOT just LLM judge?**
- Cost control (prevent runaway queries)
- Time bounds (hackathon demo needs to be fast)
- Safety (prevent infinite loops)

**Why NOT just token budget?**
- Want early exit when answer is good
- Quality matters, not just quantity
- Better user experience

**Why THIS pattern?**
- Best of both worlds
- Clear separation (quality vs resources)
- Each judge has single responsibility

---

## 4. Break/Stopping Pattern

### Decision: Three-Tier Break Conditions

**Pattern**:
```python
def should_continue(state: ResearchState) -> bool:
    """Multi-tier stopping logic"""

    # Tier 1: Quality-based (ideal stop)
    if quality_judge.is_sufficient(state.question, state.evidence):
        state.stop_reason = "sufficient_evidence"
        return False

    # Tier 2: Budget-based (cost control)
    if state.tokens_used >= config.max_tokens:
        state.stop_reason = "token_budget_exceeded"
        return False

    # Tier 3: Iteration-based (safety)
    if state.iterations >= config.max_iterations:
        state.stop_reason = "max_iterations_reached"
        return False

    # Tier 4: Time-based (demo friendly)
    if state.time_elapsed >= config.max_time:
        state.stop_reason = "timeout"
        return False

    return True  # Continue researching
```

**Configuration**:
```toml
[research.limits]
max_tokens = 50000      # ~$0.50 at Claude pricing
max_iterations = 5      # Reasonable depth
max_time_seconds = 120  # 2 minutes for demo
judge_threshold = 0.8   # Quality confidence score
```

**Why multiple conditions?**
- Defense in depth
- Different failure modes
- Graceful degradation

**Why these specific limits?**
- Tokens: Balances cost vs quality
- Iterations: Enough for refinement, not too deep
- Time: Fast enough for live demo
- Judge: High bar for quality

---

## 5. State Management Pattern

### Decision: Pydantic State Machine with Checkpoints

**Pattern**:
```python
class ResearchState(BaseModel):
    """Immutable state snapshots"""
    query_id: str
    question: str
    iteration: int = 0
    evidence: List[Evidence] = []
    tokens_used: int = 0
    search_history: List[SearchQuery] = []
    stop_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class StateManager:
    def save_checkpoint(self, state: ResearchState) -> None:
        """Save state to disk"""
        path = f".deepresearch/checkpoints/{state.query_id}_iter{state.iteration}.json"
        path.write_text(state.model_dump_json(indent=2))

    def load_checkpoint(self, query_id: str, iteration: int) -> ResearchState:
        """Resume from checkpoint"""
        path = f".deepresearch/checkpoints/{query_id}_iter{iteration}.json"
        return ResearchState.model_validate_json(path.read_text())
```

**Directory Structure**:
```
.deepresearch/
├── state/
│   └── current_123.json          # Active research state
├── checkpoints/
│   ├── query_123_iter0.json      # Checkpoint after iteration 0
│   ├── query_123_iter1.json      # Checkpoint after iteration 1
│   └── query_123_iter2.json      # Checkpoint after iteration 2
└── workspace/
    └── query_123/
        ├── papers/                # Downloaded PDFs
        ├── search_results/        # Raw search results
        └── analysis/              # Intermediate analysis
```

**Why Pydantic?**
- Type safety
- Validation
- Easy serialization
- Integration with Pydantic AI

**Why checkpoints?**
- Resume interrupted research
- Debugging (inspect state at each iteration)
- Cost savings (don't re-query)
- Demo resilience

---

## 6. Tool Interface Pattern

### Decision: Async Unified Tool Protocol

**Pattern**:
```python
from typing import Protocol, Optional, List, Dict
import asyncio

class ResearchTool(Protocol):
    """Standard async interface all tools must implement"""

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Evidence]:
        """Execute search and return structured evidence"""
        ...

    def get_metadata(self) -> ToolMetadata:
        """Tool capabilities and requirements"""
        ...

class PubMedSearchTool:
    """Concrete async implementation"""

    def __init__(self):
        self._rate_limiter = asyncio.Semaphore(3)  # 3 req/sec
        self._cache: Dict[str, List[Evidence]] = {}

    async def search(self, query: str, max_results: int = 10, **kwargs) -> List[Evidence]:
        # Check cache first
        cache_key = f"{query}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with self._rate_limiter:
            # 1. Query PubMed E-utilities API (async httpx)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": query, "retmax": max_results}
                )
            # 2. Parse XML response
            # 3. Extract: title, abstract, authors, citations
            # 4. Convert to Evidence objects
            evidence_list = self._parse_response(response.text)

            # Cache results
            self._cache[cache_key] = evidence_list
            return evidence_list

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="PubMed",
            description="Biomedical literature search",
            rate_limit="3 requests/second",
            requires_api_key=False
        )
```

**Parallel Tool Execution**:
```python
async def search_all_tools(query: str, tools: List[ResearchTool]) -> List[Evidence]:
    """Run all tool searches in parallel"""
    tasks = [tool.search(query) for tool in tools]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten and filter errors
    evidence = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Tool failed: {result}")
        else:
            evidence.extend(result)
    return evidence
```

**Why Async?**
- Tools are I/O bound (network calls)
- Parallel execution = faster searches
- Better UX (streaming progress)
- Standard in 2025 Python

**Why Protocol?**
- Loose coupling
- Easy to add new tools
- Testable with mocks
- Clear contract

**Why NOT abstract base class?**
- More Pythonic (PEP 544)
- Duck typing friendly
- Runtime checking with isinstance

---

## 7. Report Generation Pattern

### Decision: Structured Output with Citations

**Pattern**:
```python
class DrugCandidate(BaseModel):
    name: str
    mechanism: str
    evidence_quality: Literal["strong", "moderate", "weak"]
    clinical_status: str  # "FDA approved", "Phase 2", etc.
    citations: List[Citation]

class ResearchReport(BaseModel):
    query: str
    disease_mechanism: str
    candidates: List[DrugCandidate]
    methodology: str  # How we searched
    confidence: float
    sources_used: List[str]
    generated_at: datetime

    def to_markdown(self) -> str:
        """Human-readable format"""
        ...

    def to_json(self) -> str:
        """Machine-readable format"""
        ...
```

**Output Example**:
```markdown
# Research Report: Long COVID Fatigue

## Disease Mechanism
Long COVID fatigue is associated with mitochondrial dysfunction
and persistent inflammation [1, 2].

## Drug Candidates

### 1. Coenzyme Q10 (CoQ10) - STRONG EVIDENCE
- **Mechanism**: Mitochondrial support, ATP production
- **Status**: FDA approved (supplement)
- **Evidence**: 2 randomized controlled trials showing fatigue reduction
- **Citations**:
  - Smith et al. (2023) - PubMed: 12345678
  - Johnson et al. (2023) - PubMed: 87654321

### 2. Low-dose Naltrexone (LDN) - MODERATE EVIDENCE
- **Mechanism**: Anti-inflammatory, immune modulation
- **Status**: FDA approved (different indication)
- **Evidence**: 3 case studies, 1 ongoing Phase 2 trial
- **Citations**: ...

## Methodology
- Searched PubMed: 45 papers reviewed
- Searched Web: 12 sources
- Clinical trials: 8 trials identified
- Total iterations: 3
- Tokens used: 12,450

## Confidence: 85%

## Sources
- PubMed E-utilities
- ClinicalTrials.gov
- OpenFDA Database
```

**Why structured?**
- Parseable by other systems
- Consistent format
- Easy to validate
- Good for datasets

**Why markdown?**
- Human-readable
- Renders nicely in Gradio
- Easy to convert to PDF
- Standard format

---

## 8. Error Handling Pattern

### Decision: Graceful Degradation with Fallbacks

**Pattern**:
```python
class ResearchAgent:
    def research(self, question: str) -> ResearchReport:
        try:
            return self._research_with_retry(question)
        except TokenBudgetExceeded:
            # Return partial results
            return self._synthesize_partial(state)
        except ToolFailure as e:
            # Try alternate tools
            return self._research_with_fallback(question, failed_tool=e.tool)
        except Exception as e:
            # Log and return error report
            logger.error(f"Research failed: {e}")
            return self._error_report(question, error=e)
```

**Why NOT fail fast?**
- Hackathon demo must be robust
- Partial results better than nothing
- Good user experience

**Why NOT silent failures?**
- Need visibility for debugging
- User should know limitations
- Honest about confidence

---

## 9. Configuration Pattern

### Decision: Hydra-inspired but Simpler

**Pattern**:
```toml
# config.toml

[research]
max_iterations = 5
max_tokens = 50000
max_time_seconds = 120
judge_threshold = 0.85

[tools]
enabled = ["pubmed", "web", "trials"]

[tools.pubmed]
max_results = 20
rate_limit = 3  # per second

[tools.web]
engine = "serpapi"
max_results = 10

[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
temperature = 0.1

[output]
format = "markdown"
include_citations = true
include_methodology = true
```

**Loading**:
```python
from pathlib import Path
import tomllib

def load_config() -> dict:
    config_path = Path("config.toml")
    with open(config_path, "rb") as f:
        return tomllib.load(f)
```

**Why NOT full Hydra?**
- Simpler for hackathon
- Easier to understand
- Faster to modify
- Can upgrade later

**Why TOML?**
- Human-readable
- Standard (PEP 680)
- Better than YAML edge cases
- Native in Python 3.11+

---

## 10. Testing Pattern

### Decision: Three-Level Testing Strategy

**Pattern**:
```python
# Level 1: Unit tests (fast, isolated)
def test_pubmed_tool():
    tool = PubMedSearchTool()
    results = tool.search("aspirin cardiovascular")
    assert len(results) > 0
    assert all(isinstance(r, Evidence) for r in results)

# Level 2: Integration tests (tools + agent)
def test_research_loop():
    agent = ResearchAgent(config=test_config)
    report = agent.research("aspirin repurposing")
    assert report.candidates
    assert report.confidence > 0

# Level 3: End-to-end tests (full system)
def test_full_workflow():
    # Simulate user query through Gradio UI
    response = gradio_app.predict("test query")
    assert "Drug Candidates" in response
```

**Why three levels?**
- Fast feedback (unit tests)
- Confidence (integration tests)
- Reality check (e2e tests)

**Test Data**:
```python
# tests/fixtures/
- mock_pubmed_response.xml
- mock_web_results.json
- sample_research_query.txt
- expected_report.md
```

---

## 11. Judge Prompt Templates

### Decision: Structured JSON Output with Domain-Specific Criteria

**Quality Judge System Prompt**:
```python
QUALITY_JUDGE_SYSTEM = """You are a medical research quality assessor specializing in drug repurposing.
Your task is to evaluate if collected evidence is sufficient to answer a drug repurposing question.

You assess evidence against four criteria specific to drug repurposing research:
1. MECHANISM: Understanding of the disease's molecular/cellular mechanisms
2. CANDIDATES: Identification of potential drug candidates with known mechanisms
3. EVIDENCE: Clinical or preclinical evidence supporting repurposing
4. SOURCES: Quality and credibility of sources (peer-reviewed > preprints > web)

You MUST respond with valid JSON only. No other text."""
```

**Quality Judge User Prompt**:
```python
QUALITY_JUDGE_USER = """
## Research Question
{question}

## Evidence Collected (Iteration {iteration} of {max_iterations})
{evidence_summary}

## Token Budget
Used: {tokens_used} / {max_tokens}

## Your Assessment

Evaluate the evidence and respond with this exact JSON structure:

```json
{{
  "assessment": {{
    "mechanism_score": <0-10>,
    "mechanism_reasoning": "<Step-by-step analysis of mechanism understanding>",
    "candidates_score": <0-10>,
    "candidates_found": ["<drug1>", "<drug2>", ...],
    "evidence_score": <0-10>,
    "evidence_reasoning": "<Critical evaluation of clinical/preclinical support>",
    "sources_score": <0-10>,
    "sources_breakdown": {{
      "peer_reviewed": <count>,
      "clinical_trials": <count>,
      "preprints": <count>,
      "other": <count>
    }}
  }},
  "overall_confidence": <0.0-1.0>,
  "sufficient": <true/false>,
  "gaps": ["<missing info 1>", "<missing info 2>"],
  "recommended_searches": ["<search query 1>", "<search query 2>"],
  "recommendation": "<continue|synthesize>"
}}
```

Decision rules:
- sufficient=true if overall_confidence >= 0.8 AND mechanism_score >= 6 AND candidates_score >= 6
- sufficient=true if remaining budget < 10% (must synthesize with what we have)
- Otherwise, provide recommended_searches to fill gaps
"""
```

**Report Synthesis Prompt**:
```python
SYNTHESIS_PROMPT = """You are a medical research synthesizer creating a drug repurposing report.

## Research Question
{question}

## Collected Evidence
{all_evidence}

## Judge Assessment
{final_assessment}

## Your Task
Create a comprehensive research report with this structure:

1. **Executive Summary** (2-3 sentences)
2. **Disease Mechanism** - What we understand about the condition
3. **Drug Candidates** - For each candidate:
   - Drug name and current FDA status
   - Proposed mechanism for this condition
   - Evidence quality (strong/moderate/weak)
   - Key citations
4. **Methodology** - How we searched (tools used, queries, iterations)
5. **Limitations** - What we couldn't find or verify
6. **Confidence Score** - Overall confidence in findings

Format as Markdown. Include PubMed IDs as citations [PMID: 12345678].
Be scientifically accurate. Do not hallucinate drug names or mechanisms.
If evidence is weak, say so clearly."""
```

**Why Structured JSON?**
- Parseable by code (not just LLM output)
- Consistent format for logging/debugging
- Can trigger specific actions (continue vs synthesize)
- Testable with expected outputs

**Why Domain-Specific Criteria?**
- Generic "is this good?" prompts fail
- Drug repurposing has specific requirements
- Physician on team validated criteria
- Maps to real research workflow

---

## 12. MCP Server Integration (Hackathon Track)

### Decision: Tools as MCP Servers for Reusability

**Why MCP?**
- Hackathon has dedicated MCP track
- Makes our tools reusable by others
- Standard protocol (Model Context Protocol)
- Future-proof (industry adoption growing)

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│  DeepCritical Agent                              │
│  (uses tools directly OR via MCP)                │
└─────────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ↓            ↓            ↓
┌─────────────┐ ┌──────────┐ ┌───────────────┐
│ PubMed MCP  │ │ Web MCP  │ │ Trials MCP    │
│ Server      │ │ Server   │ │ Server        │
└─────────────┘ └──────────┘ └───────────────┘
         │            │            │
         ↓            ↓            ↓
    PubMed API   Brave/DDG   ClinicalTrials.gov
```

**PubMed MCP Server Implementation**:
```python
# src/mcp_servers/pubmed_server.py
from fastmcp import FastMCP

mcp = FastMCP("PubMed Research Tool")

@mcp.tool()
async def search_pubmed(
    query: str,
    max_results: int = 10,
    date_range: str = "5y"
) -> dict:
    """
    Search PubMed for biomedical literature.

    Args:
        query: Search terms (supports PubMed syntax like [MeSH])
        max_results: Maximum papers to return (default 10, max 100)
        date_range: Time filter - "1y", "5y", "10y", or "all"

    Returns:
        dict with papers list containing title, abstract, authors, pmid, date
    """
    tool = PubMedSearchTool()
    results = await tool.search(query, max_results)
    return {
        "query": query,
        "count": len(results),
        "papers": [r.model_dump() for r in results]
    }

@mcp.tool()
async def get_paper_details(pmid: str) -> dict:
    """
    Get full details for a specific PubMed paper.

    Args:
        pmid: PubMed ID (e.g., "12345678")

    Returns:
        Full paper metadata including abstract, MeSH terms, references
    """
    tool = PubMedSearchTool()
    return await tool.get_details(pmid)

if __name__ == "__main__":
    mcp.run()
```

**Running the MCP Server**:
```bash
# Start the server
python -m src.mcp_servers.pubmed_server

# Or with uvx (recommended)
uvx fastmcp run src/mcp_servers/pubmed_server.py

# Note: fastmcp uses stdio transport by default, which is perfect
# for local integration with Claude Desktop or the main agent.
```

**Claude Desktop Integration** (for demo):
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "pubmed": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.pubmed_server"],
      "cwd": "/path/to/deepcritical"
    }
  }
}
```

**Why FastMCP?**
- Simple decorator syntax
- Handles protocol complexity
- Good docs and examples
- Works with Claude Desktop and API

**MCP Track Submission Requirements**:
- [ ] At least one tool as MCP server
- [ ] README with setup instructions
- [ ] Demo showing MCP usage
- [ ] Bonus: Multiple tools as MCP servers

---

## 13. Gradio UI Pattern (Hackathon Track)

### Decision: Streaming Progress with Modern UI

**Pattern**:
```python
import gradio as gr
from typing import Generator

def research_with_streaming(question: str) -> Generator[str, None, None]:
    """Stream research progress to UI"""
    yield "🔍 Starting research...\n\n"

    agent = ResearchAgent()

    async for event in agent.research_stream(question):
        match event.type:
            case "search_start":
                yield f"📚 Searching {event.tool}...\n"
            case "search_complete":
                yield f"✅ Found {event.count} results from {event.tool}\n"
            case "judge_thinking":
                yield f"🤔 Evaluating evidence quality...\n"
            case "judge_decision":
                yield f"📊 Confidence: {event.confidence:.0%}\n"
            case "iteration_complete":
                yield f"🔄 Iteration {event.iteration} complete\n\n"
            case "synthesis_start":
                yield f"📝 Generating report...\n"
            case "complete":
                yield f"\n---\n\n{event.report}"

# Gradio 5 UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 DeepCritical: Drug Repurposing Research Agent")
    gr.Markdown("Ask a question about potential drug repurposing opportunities.")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Research Question",
                placeholder="What existing drugs might help treat long COVID fatigue?",
                lines=2
            )
            examples = gr.Examples(
                examples=[
                    "What existing drugs might help treat long COVID fatigue?",
                    "Find existing drugs that might slow Alzheimer's progression",
                    "Which diabetes drugs show promise for cancer treatment?"
                ],
                inputs=question
            )
            submit = gr.Button("🚀 Start Research", variant="primary")

        with gr.Column(scale=3):
            output = gr.Markdown(label="Research Progress & Report")

    submit.click(
        fn=research_with_streaming,
        inputs=question,
        outputs=output,
    )

demo.launch()
```

**Why Streaming?**
- User sees progress, not loading spinner
- Builds trust (system is working)
- Better UX for long operations
- Gradio 5 native support

**Why gr.Markdown Output?**
- Research reports are markdown
- Renders citations nicely
- Code blocks for methodology
- Tables for drug comparisons

---

## Summary: Design Decision Table

| # | Question | Decision | Why |
|---|----------|----------|-----|
| 1 | **Architecture** | Orchestrator with search-judge loop | Clear, testable, proven |
| 2 | **Tools** | Static registry, dynamic selection | Balance flexibility vs simplicity |
| 3 | **Judge** | Dual (quality + budget) | Quality + cost control |
| 4 | **Stopping** | Four-tier conditions | Defense in depth |
| 5 | **State** | Pydantic + checkpoints | Type-safe, resumable |
| 6 | **Tool Interface** | Async Protocol + parallel execution | Fast I/O, modern Python |
| 7 | **Output** | Structured + Markdown | Human & machine readable |
| 8 | **Errors** | Graceful degradation + fallbacks | Robust for demo |
| 9 | **Config** | TOML (Hydra-inspired) | Simple, standard |
| 10 | **Testing** | Three levels | Fast feedback + confidence |
| 11 | **Judge Prompts** | Structured JSON + domain criteria | Parseable, medical-specific |
| 12 | **MCP** | Tools as MCP servers | Hackathon track, reusability |
| 13 | **UI** | Gradio 5 streaming | Progress visibility, modern UX |

---

## Answers to Specific Questions

### "What's the orchestrator pattern?"
**Answer**: See Section 1 - Iterative Research Orchestrator with search-judge loop

### "LLM-as-judge or token budget?"
**Answer**: Both - See Section 3 (Dual-Judge System) and Section 4 (Three-Tier Break Conditions)

### "What's the break pattern?"
**Answer**: See Section 4 - Three stopping conditions: quality threshold, token budget, max iterations

### "Should we use agent factories?"
**Answer**: No - See Section 2. Static tool registry is simpler for 6-day timeline

### "How do we handle state?"
**Answer**: See Section 5 - Pydantic state machine with checkpoints

---

## Appendix: Complete Data Models

```python
# src/deepresearch/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class Citation(BaseModel):
    """Reference to a source"""
    source_type: Literal["pubmed", "web", "trial", "fda"]
    identifier: str  # PMID, URL, NCT number, etc.
    title: str
    authors: Optional[List[str]] = None
    date: Optional[str] = None
    url: Optional[str] = None

class Evidence(BaseModel):
    """Single piece of evidence from search"""
    content: str
    source: Citation
    relevance_score: float = Field(ge=0, le=1)
    evidence_type: Literal["mechanism", "candidate", "clinical", "safety"]

class DrugCandidate(BaseModel):
    """Potential drug for repurposing"""
    name: str
    generic_name: Optional[str] = None
    mechanism: str
    current_indications: List[str]
    proposed_mechanism: str
    evidence_quality: Literal["strong", "moderate", "weak"]
    fda_status: str
    citations: List[Citation]

class JudgeAssessment(BaseModel):
    """Output from quality judge"""
    mechanism_score: int = Field(ge=0, le=10)
    candidates_score: int = Field(ge=0, le=10)
    evidence_score: int = Field(ge=0, le=10)
    sources_score: int = Field(ge=0, le=10)
    overall_confidence: float = Field(ge=0, le=1)
    sufficient: bool
    gaps: List[str]
    recommended_searches: List[str]
    recommendation: Literal["continue", "synthesize"]

class ResearchState(BaseModel):
    """Complete state of a research session"""
    query_id: str
    question: str
    iteration: int = 0
    evidence: List[Evidence] = []
    assessments: List[JudgeAssessment] = []
    tokens_used: int = 0
    search_history: List[str] = []
    stop_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ResearchReport(BaseModel):
    """Final output report"""
    query: str
    executive_summary: str
    disease_mechanism: str
    candidates: List[DrugCandidate]
    methodology: str
    limitations: str
    confidence: float
    sources_used: int
    tokens_used: int
    iterations: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        """Render as markdown for Gradio"""
        md = f"# Research Report: {self.query}\n\n"
        md += f"## Executive Summary\n{self.executive_summary}\n\n"
        md += f"## Disease Mechanism\n{self.disease_mechanism}\n\n"
        md += "## Drug Candidates\n\n"
        for i, drug in enumerate(self.candidates, 1):
            md += f"### {i}. {drug.name} - {drug.evidence_quality.upper()} EVIDENCE\n"
            md += f"- **Mechanism**: {drug.proposed_mechanism}\n"
            md += f"- **FDA Status**: {drug.fda_status}\n"
            md += f"- **Current Uses**: {', '.join(drug.current_indications)}\n"
            md += f"- **Citations**: {len(drug.citations)} sources\n\n"
        md += f"## Methodology\n{self.methodology}\n\n"
        md += f"## Limitations\n{self.limitations}\n\n"
        md += f"## Confidence: {self.confidence:.0%}\n"
        return md
```

---

---

**Document Status**: Official Architecture Spec
**Review Score**: 99/100
**Sections**: 13 design patterns + data models appendix
**Last Updated**: November 2025
