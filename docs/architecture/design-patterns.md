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
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │  Tool Coordinator             │  │
│  │  - PubMed Search              │  │
│  │  - Web Search                 │  │
│  │  - Clinical Trials            │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │  Evidence Aggregator          │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │  Quality Judge                │  │
│  │  (LLM-based assessment)       │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│       Loop or Synthesize?           │
│              ↓                      │
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
│  DeepBoner Agent                             │
│  (uses tools directly OR via MCP)               │
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
      "cwd": "/path/to/deepboner"
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
    gr.Markdown("# 🔬 DeepBoner: Drug Repurposing Research Agent")
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

## 14. Alternative Frameworks Considered

We researched major agent frameworks before settling on our stack. Here's why we chose what we chose, and what we'd steal if we're shipping like animals and have time for Gucci upgrades.

### Frameworks Evaluated

| Framework | Repo | What It Does |
|-----------|------|--------------|
| **Microsoft AutoGen** | [github.com/microsoft/autogen](https://github.com/microsoft/autogen) | Multi-agent orchestration, complex workflows |
| **Claude Agent SDK** | [github.com/anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python) | Anthropic's official agent framework |
| **Pydantic AI** | [github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | Type-safe agents, structured outputs |

### Why NOT AutoGen (Microsoft)?

**Pros:**
- Battle-tested multi-agent orchestration
- `reflect_on_tool_use` - model reviews its own tool results
- `max_tool_iterations` - built-in iteration limits
- Concurrent tool execution
- Rich ecosystem (AutoGen Studio, benchmarks)

**Cons for MVP:**
- Heavy dependency tree (50+ packages)
- Complex configuration (YAML + Python)
- Overkill for single-agent search-judge loop
- Learning curve eats into 6-day timeline

**Verdict:** Great for multi-agent systems. Overkill for our MVP.

### Why NOT Claude Agent SDK (Anthropic)?

**Pros:**
- Official Anthropic framework
- Clean `@tool` decorator pattern
- In-process MCP servers (no subprocess)
- Hooks for pre/post tool execution
- Direct Claude Code integration

**Cons for MVP:**
- Requires Claude Code CLI bundled
- Node.js dependency for some features
- Designed for Claude Code ecosystem, not standalone agents
- Less flexible for custom LLM providers

**Verdict:** Would be great if we were building ON Claude Code. We're building a standalone agent.

### Why Pydantic AI + FastMCP (Our Choice)

**Pros:**
- ✅ Simple, Pythonic API
- ✅ Native async/await
- ✅ Type-safe with Pydantic
- ✅ Works with any LLM provider
- ✅ FastMCP for clean MCP servers
- ✅ Minimal dependencies
- ✅ Can ship MVP in 6 days

**Cons:**
- Newer framework (less battle-tested)
- Smaller ecosystem
- May need to build more from scratch

**Verdict:** Right tool for the job. Ship fast, iterate later.

---

## 15. Stretch Goals: Gucci Bangers (If We're Shipping Like Animals)

If MVP ships early and we're crushing it, here's what we'd steal from other frameworks:

### Tier 1: Quick Wins (2-4 hours each)

#### From Claude Agent SDK: `@tool` Decorator Pattern
Replace our Protocol-based tools with cleaner decorators:

```python
# CURRENT (Protocol-based)
class PubMedSearchTool:
    async def search(self, query: str, max_results: int = 10) -> List[Evidence]:
        ...

# UPGRADE (Decorator-based, stolen from Claude SDK)
from claude_agent_sdk import tool

@tool("search_pubmed", "Search PubMed for biomedical papers", {
    "query": str,
    "max_results": int
})
async def search_pubmed(args):
    results = await _do_pubmed_search(args["query"], args["max_results"])
    return {"content": [{"type": "text", "text": json.dumps(results)}]}
```

**Why it's Gucci:** Cleaner syntax, automatic schema generation, less boilerplate.

#### From AutoGen: Reflect on Tool Use
Add a reflection step where the model reviews its own tool results:

```python
# CURRENT: Judge evaluates evidence
assessment = await judge.assess(question, evidence)

# UPGRADE: Add reflection step (stolen from AutoGen)
class ReflectiveJudge:
    async def assess_with_reflection(self, question, evidence, tool_results):
        # First pass: raw assessment
        initial = await self._assess(question, evidence)

        # Reflection: "Did I use the tools correctly?"
        reflection = await self._reflect_on_tool_use(tool_results)

        # Final: combine assessment + reflection
        return self._combine(initial, reflection)
```

**Why it's Gucci:** Catches tool misuse, improves accuracy, more robust judge.

### Tier 2: Medium Lifts (4-8 hours each)

#### From AutoGen: Concurrent Tool Execution
Run multiple tools in parallel with proper error handling:

```python
# CURRENT: Sequential with asyncio.gather
results = await asyncio.gather(*[tool.search(query) for tool in tools])

# UPGRADE: AutoGen-style with cancellation + timeout
from autogen_core import CancellationToken

async def execute_tools_concurrent(tools, query, timeout=30):
    token = CancellationToken()

    async def run_with_timeout(tool):
        try:
            return await asyncio.wait_for(
                tool.search(query, cancellation_token=token),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            token.cancel()  # Cancel other tools
            return ToolError(f"{tool.name} timed out")

    return await asyncio.gather(*[run_with_timeout(t) for t in tools])
```

**Why it's Gucci:** Proper timeout handling, cancellation propagation, production-ready.

#### From Claude SDK: Hooks System
Add pre/post hooks for logging, validation, cost tracking:

```python
# UPGRADE: Hook system (stolen from Claude SDK)
class HookManager:
    async def pre_tool_use(self, tool_name, args):
        """Called before every tool execution"""
        logger.info(f"Calling {tool_name} with {args}")
        self.cost_tracker.start_timer()

    async def post_tool_use(self, tool_name, result, duration):
        """Called after every tool execution"""
        self.cost_tracker.record(tool_name, duration)
        if result.is_error:
            self.error_tracker.record(tool_name, result.error)
```

**Why it's Gucci:** Observability, debugging, cost tracking, production-ready.

### Tier 3: Big Lifts (Post-Hackathon)

#### Full AutoGen Integration
If we want multi-agent capabilities later:

```python
# POST-HACKATHON: Multi-agent drug repurposing
from autogen_agentchat import AssistantAgent, GroupChat

literature_agent = AssistantAgent(
    name="LiteratureReviewer",
    tools=[pubmed_search, web_search],
    system_message="You search and summarize medical literature."
)

mechanism_agent = AssistantAgent(
    name="MechanismAnalyzer",
    tools=[pathway_db, protein_db],
    system_message="You analyze disease mechanisms and drug targets."
)

synthesis_agent = AssistantAgent(
    name="ReportSynthesizer",
    system_message="You synthesize findings into actionable reports."
)

# Orchestrate multi-agent workflow
group_chat = GroupChat(
    agents=[literature_agent, mechanism_agent, synthesis_agent],
    max_round=10
)
```

**Why it's Gucci:** True multi-agent collaboration, specialized roles, scalable.

---

## Priority Order for Stretch Goals

| Priority | Feature | Source | Effort | Impact |
|----------|---------|--------|--------|--------|
| 1 | `@tool` decorator | Claude SDK | 2 hrs | High - cleaner code |
| 2 | Reflect on tool use | AutoGen | 3 hrs | High - better accuracy |
| 3 | Hooks system | Claude SDK | 4 hrs | Medium - observability |
| 4 | Concurrent + cancellation | AutoGen | 4 hrs | Medium - robustness |
| 5 | Multi-agent | AutoGen | 8+ hrs | Post-hackathon |

---

## The Bottom Line

```
┌─────────────────────────────────────────────────────────────┐
│  MVP (Days 1-4): Pydantic AI + FastMCP                      │
│  - Ship working drug repurposing agent                      │
│  - Search-judge loop with PubMed + Web                      │
│  - Gradio UI with streaming                                 │
│  - MCP server for hackathon track                           │
├─────────────────────────────────────────────────────────────┤
│  If Crushing It (Days 5-6): Steal the Gucci                 │
│  - @tool decorators from Claude SDK                         │
│  - Reflect on tool use from AutoGen                         │
│  - Hooks for observability                                  │
├─────────────────────────────────────────────────────────────┤
│  Post-Hackathon: Full AutoGen Integration                   │
│  - Multi-agent workflows                                    │
│  - Specialized agent roles                                  │
│  - Production-grade orchestration                           │
└─────────────────────────────────────────────────────────────┘
```

**Ship MVP first. Steal bangers if time. Scale later.**

---

## 16. Reference Implementation Resources

We've cloned production-ready repos into `reference_repos/` that we can vendor, copy from, or just USE directly. This section documents what's available and how to leverage it.

### Cloned Repositories

| Repository | Location | What It Provides |
|------------|----------|------------------|
| **pydanticai-research-agent** | `reference_repos/pydanticai-research-agent/` | Complete PydanticAI agent with Brave Search |
| **pubmed-mcp-server** | `reference_repos/pubmed-mcp-server/` | Production-grade PubMed MCP server (TypeScript) |
| **autogen-microsoft** | `reference_repos/autogen-microsoft/` | Microsoft's multi-agent framework |
| **claude-agent-sdk** | `reference_repos/claude-agent-sdk/` | Anthropic's agent SDK with @tool decorator |

### 🔥 CHEAT CODE: Production PubMed MCP Already Exists

The `pubmed-mcp-server` is **production-grade** and has EVERYTHING we need:

```bash
# Already available tools in pubmed-mcp-server:
pubmed_search_articles    # Search PubMed with filters, date ranges
pubmed_fetch_contents     # Get full article details by PMID
pubmed_article_connections # Find citations, related articles
pubmed_research_agent     # Generate research plan outlines
pubmed_generate_chart     # Create PNG charts from data
```

**Option 1: Use it directly via npx**
```json
{
  "mcpServers": {
    "pubmed": {
      "command": "npx",
      "args": ["@cyanheads/pubmed-mcp-server"],
      "env": { "NCBI_API_KEY": "your_key" }
    }
  }
}
```

**Option 2: Vendor the logic into Python**
The TypeScript code in `reference_repos/pubmed-mcp-server/src/` shows exactly how to:
- Construct PubMed E-utilities queries
- Handle rate limiting (3/sec without key, 10/sec with key)
- Parse XML responses
- Extract article metadata

### PydanticAI Research Agent Patterns

The `pydanticai-research-agent` repo provides copy-paste patterns:

**Agent Definition** (`agents/research_agent.py`):
```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class ResearchAgentDependencies:
    brave_api_key: str
    session_id: Optional[str] = None

research_agent = Agent(
    get_llm_model(),
    deps_type=ResearchAgentDependencies,
    system_prompt=SYSTEM_PROMPT
)

@research_agent.tool
async def search_web(
    ctx: RunContext[ResearchAgentDependencies],
    query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """Search with context access via ctx.deps"""
    results = await search_web_tool(ctx.deps.brave_api_key, query, max_results)
    return results
```

**Brave Search Tool** (`tools/brave_search.py`):
```python
async def search_web_tool(api_key: str, query: str, count: int = 10) -> List[Dict]:
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params={"q": query, "count": count},
            timeout=30.0
        )
    # Handle 429 rate limit, 401 auth errors
    data = response.json()
    return data.get("web", {}).get("results", [])
```

**Pydantic Models** (`models/research_models.py`):
```python
class BraveSearchResult(BaseModel):
    title: str
    url: str
    description: str
    score: float = Field(ge=0.0, le=1.0)
```

### Microsoft Agent Framework Orchestration Patterns

From [deepwiki.com/microsoft/agent-framework](https://deepwiki.com/microsoft/agent-framework/3.4-workflows-and-orchestration):

#### Sequential Orchestration
```
Agent A → Agent B → Agent C (each receives prior outputs)
```
**Use when:** Tasks have dependencies, results inform next steps.

#### Concurrent (Fan-out/Fan-in)
```
           ┌→ Agent A ─┐
Dispatcher ├→ Agent B ─┼→ Aggregator
           └→ Agent C ─┘
```
**Use when:** Independent tasks can run in parallel, results need consolidation.
**Our use:** Parallel PubMed + Web search.

#### Handoff Orchestration
```
Coordinator → routes to → Specialist A, B, or C based on request
```
**Use when:** Router decides which search strategy based on query type.
**Our use:** Route "mechanism" vs "clinical trial" vs "drug info" queries.

#### HITL (Human-in-the-Loop)
```
Agent → RequestInfoEvent → Human validates → Agent continues
```
**Use when:** Critical judgment points need human validation.
**Our use:** Optional "approve drug candidates before synthesis" step.

### Recommended Hybrid Pattern for Our Agent

Based on all the research, here's our recommended implementation:

```
┌─────────────────────────────────────────────────────────┐
│  1. ROUTER (Handoff Pattern)                             │
│     - Analyze query type                                 │
│     - Choose search strategy                             │
├─────────────────────────────────────────────────────────┤
│  2. SEARCH (Concurrent Pattern)                          │
│     - Fan-out to PubMed + Web in parallel                │
│     - Timeout handling per AutoGen patterns              │
│     - Aggregate results                                  │
├─────────────────────────────────────────────────────────┤
│  3. JUDGE (Sequential + Budget)                          │
│     - Quality assessment                                 │
│     - Token/iteration budget check                       │
│     - Recommend: continue or synthesize                  │
├─────────────────────────────────────────────────────────┤
│  4. SYNTHESIZE (Final Agent)                             │
│     - Generate research report                           │
│     - Include citations                                  │
│     - Stream to Gradio UI                                │
└─────────────────────────────────────────────────────────┘
```

### Quick Start: Minimal Implementation Path

**Day 1-2: Core Loop**
1. Copy `search_web_tool` from `pydanticai-research-agent/tools/brave_search.py`
2. Implement PubMed search (reference `pubmed-mcp-server/src/` for E-utilities patterns)
3. Wire up basic search-judge loop

**Day 3: Judge + State**
1. Implement quality judge with JSON structured output
2. Add budget judge
3. Add Pydantic state management

**Day 4: UI + MCP**
1. Gradio streaming UI
2. Wrap PubMed tool as FastMCP server

**Day 5-6: Polish + Deploy**
1. HuggingFace Spaces deployment
2. Demo video
3. Stretch goals if time

---

## 17. External Resources & MCP Servers

### Available PubMed MCP Servers (Community)

| Server | Author | Features | Link |
|--------|--------|----------|------|
| **pubmed-mcp-server** | cyanheads | Full E-utilities, research agent, charts | [GitHub](https://github.com/cyanheads/pubmed-mcp-server) |
| **BioMCP** | GenomOncology | PubMed + ClinicalTrials + MyVariant | [GitHub](https://github.com/genomoncology/biomcp) |
| **PubMed-MCP-Server** | JackKuo666 | Basic search, metadata access | [GitHub](https://github.com/JackKuo666/PubMed-MCP-Server) |

### Web Search Options

| Tool | Free Tier | API Key | Async Support |
|------|-----------|---------|---------------|
| **Brave Search** | 2000/month | Required | Yes (httpx) |
| **DuckDuckGo** | Unlimited | No | Yes (duckduckgo-search) |
| **SerpAPI** | None | Required | Yes |

**Recommended:** Start with DuckDuckGo (free, no key), upgrade to Brave for production.

```python
# DuckDuckGo async search (no API key needed!)
from duckduckgo_search import DDGS

async def search_ddg(query: str, max_results: int = 10) -> List[Dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "url": r["href"], "description": r["body"]} for r in results]
```

---

**Document Status**: Official Architecture Spec
**Review Score**: 100/100 (Ironclad Gucci Banger Edition)
**Sections**: 17 design patterns + data models appendix + reference repos + stretch goals
**Last Updated**: November 2025
