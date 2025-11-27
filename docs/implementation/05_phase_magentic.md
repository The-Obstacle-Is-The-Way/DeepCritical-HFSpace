# Phase 5 Implementation Spec: Magentic Integration

**Goal**: Upgrade orchestrator to use Microsoft Agent Framework's Magentic-One pattern.
**Philosophy**: "Same API, Better Engine."
**Prerequisite**: Phase 4 complete (MVP working end-to-end)

---

## 1. Why Magentic?

Magentic-One provides:
- **LLM-powered manager** that dynamically plans, selects agents, tracks progress
- **Built-in stall detection** and automatic replanning
- **Checkpointing** for pause/resume workflows
- **Event streaming** for real-time UI updates
- **Multi-agent coordination** with round limits and reset logic

---

## 2. Critical Architecture Understanding

### 2.1 How Magentic Actually Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MagenticBuilder Workflow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Task: "Research drug repurposing for metformin alzheimer"          │
│                              ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   StandardMagenticManager                         │   │
│  │                                                                   │   │
│  │  1. plan() → LLM generates facts & plan                          │   │
│  │  2. create_progress_ledger() → LLM decides:                      │   │
│  │     - is_request_satisfied?                                       │   │
│  │     - next_speaker: "searcher"                                    │   │
│  │     - instruction_or_question: "Search for clinical trials..."   │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│           NATURAL LANGUAGE INSTRUCTION sent to agent                     │
│           "Search for clinical trials about metformin..."                │
│                              ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      ChatAgent (searcher)                         │   │
│  │                                                                   │   │
│  │  chat_client (INTERNAL LLM) ← understands instruction            │   │
│  │         ↓                                                         │   │
│  │  "I'll search for metformin alzheimer clinical trials"           │   │
│  │         ↓                                                         │   │
│  │  tools=[search_pubmed, search_clinicaltrials] ← calls tools      │   │
│  │         ↓                                                         │   │
│  │  Returns natural language response to manager                     │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                           │
│                    Manager evaluates response                            │
│                    Decides next agent or completion                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Critical Insight

**Microsoft's ChatAgent has an INTERNAL LLM (`chat_client`) that:**
1. Receives natural language instructions from the manager
2. Understands what action to take
3. Calls attached tools (functions)
4. Returns natural language responses

**Our previous implementation was WRONG because:**
- We wrapped handlers as bare `BaseAgent` subclasses
- No internal LLM to understand instructions
- Raw instruction text was passed directly to APIs (PubMed doesn't understand "Search for clinical trials...")

### 2.3 Correct Pattern: ChatAgent with Tools

```python
# CORRECT: Agent backed by LLM that calls tools
from agent_framework import ChatAgent, AIFunction
from agent_framework.openai import OpenAIChatClient

# Define tool that ChatAgent can call
@AIFunction
async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature.

    Args:
        query: Search keywords (e.g., "metformin alzheimer mechanism")
        max_results: Maximum number of results to return
    """
    result = await pubmed_tool.search(query, max_results)
    return format_results(result)

# ChatAgent with internal LLM + tools
search_agent = ChatAgent(
    name="SearchAgent",
    description="Searches biomedical databases for drug repurposing evidence",
    instructions="You search PubMed, ClinicalTrials.gov, and bioRxiv for evidence.",
    chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),  # INTERNAL LLM
    tools=[search_pubmed, search_clinicaltrials, search_biorxiv],  # TOOLS
)
```

---

## 3. Correct Implementation

### 3.1 Shared State Module (`src/agents/state.py`)

**CRITICAL**: Tools must update shared state so:
1. EmbeddingService can deduplicate across searches
2. ReportAgent can access structured Evidence objects for citations

```python
"""Shared state for Magentic agents.

This module provides global state that tools update as a side effect.
ChatAgent tools return strings to the LLM, but also update this state
for semantic deduplication and structured citation access.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService

from src.utils.models import Evidence

logger = structlog.get_logger()


class MagenticState:
    """Shared state container for Magentic workflow.

    Maintains:
    - evidence_store: All collected Evidence objects (for citations)
    - embedding_service: Optional semantic search (for deduplication)
    """

    def __init__(self) -> None:
        self.evidence_store: list[Evidence] = []
        self.embedding_service: EmbeddingService | None = None
        self._seen_urls: set[str] = set()

    def init_embedding_service(self) -> None:
        """Lazy-initialize embedding service if available."""
        if self.embedding_service is not None:
            return
        try:
            from src.services.embeddings import get_embedding_service
            self.embedding_service = get_embedding_service()
            logger.info("Embedding service enabled for Magentic mode")
        except Exception as e:
            logger.warning("Embedding service unavailable", error=str(e))

    async def add_evidence(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """Add evidence with semantic deduplication.

        Args:
            evidence_list: New evidence from search

        Returns:
            List of unique evidence (not duplicates)
        """
        if not evidence_list:
            return []

        # URL-based deduplication first (fast)
        url_unique = [
            e for e in evidence_list
            if e.citation.url not in self._seen_urls
        ]

        # Semantic deduplication if available
        if self.embedding_service and url_unique:
            try:
                unique = await self.embedding_service.deduplicate(url_unique, threshold=0.85)
                logger.info(
                    "Semantic deduplication",
                    before=len(url_unique),
                    after=len(unique),
                )
            except Exception as e:
                logger.warning("Deduplication failed, using URL-based", error=str(e))
                unique = url_unique
        else:
            unique = url_unique

        # Update state
        for e in unique:
            self._seen_urls.add(e.citation.url)
            self.evidence_store.append(e)

        return unique

    async def search_related(self, query: str, n_results: int = 5) -> list[Evidence]:
        """Find semantically related evidence from vector store.

        Args:
            query: Search query
            n_results: Number of related items

        Returns:
            Related Evidence objects (reconstructed from vector store)
        """
        if not self.embedding_service:
            return []

        try:
            from src.utils.models import Citation

            related = await self.embedding_service.search_similar(query, n_results)
            evidence = []

            for item in related:
                if item["id"] in self._seen_urls:
                    continue  # Already in results

                meta = item.get("metadata", {})
                authors_str = meta.get("authors", "")
                authors = [a.strip() for a in authors_str.split(",") if a.strip()]

                ev = Evidence(
                    content=item["content"],
                    citation=Citation(
                        title=meta.get("title", "Related Evidence"),
                        url=item["id"],
                        source=meta.get("source", "pubmed"),
                        date=meta.get("date", "n.d."),
                        authors=authors,
                    ),
                    relevance=max(0.0, 1.0 - item.get("distance", 0.5)),
                )
                evidence.append(ev)

            return evidence
        except Exception as e:
            logger.warning("Related search failed", error=str(e))
            return []

    def reset(self) -> None:
        """Reset state for new workflow run."""
        self.evidence_store.clear()
        self._seen_urls.clear()


# Global singleton for workflow
_state: MagenticState | None = None


def get_magentic_state() -> MagenticState:
    """Get or create the global Magentic state."""
    global _state
    if _state is None:
        _state = MagenticState()
    return _state


def reset_magentic_state() -> None:
    """Reset state for a fresh workflow run."""
    global _state
    if _state is not None:
        _state.reset()
    else:
        _state = MagenticState()
```

### 3.2 Tool Functions (`src/agents/tools.py`)

Tools call APIs AND update shared state. Return strings to LLM, but also store structured Evidence.

```python
"""Tool functions for Magentic agents.

IMPORTANT: These tools do TWO things:
1. Return formatted strings to the ChatAgent's internal LLM
2. Update shared state (evidence_store, embeddings) as a side effect

This preserves semantic deduplication and structured citation access.
"""
from agent_framework import AIFunction

from src.agents.state import get_magentic_state
from src.tools.biorxiv import BioRxivTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool

# Singleton tool instances
_pubmed = PubMedTool()
_clinicaltrials = ClinicalTrialsTool()
_biorxiv = BioRxivTool()


def _format_results(results: list, source_name: str, query: str) -> str:
    """Format search results for LLM consumption."""
    if not results:
        return f"No {source_name} results found for: {query}"

    output = [f"Found {len(results)} {source_name} results:\n"]
    for i, r in enumerate(results[:10], 1):
        output.append(f"{i}. **{r.citation.title}**")
        output.append(f"   Source: {r.citation.source} | Date: {r.citation.date}")
        output.append(f"   {r.content[:300]}...")
        output.append(f"   URL: {r.citation.url}\n")

    return "\n".join(output)


@AIFunction
async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical research papers.

    Use this tool to find peer-reviewed scientific literature about
    drugs, diseases, mechanisms of action, and clinical studies.

    Args:
        query: Search keywords (e.g., "metformin alzheimer mechanism")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of papers with titles, abstracts, and citations
    """
    # 1. Execute search
    results = await _pubmed.search(query, max_results)

    # 2. Update shared state (semantic dedup + evidence store)
    state = get_magentic_state()
    unique = await state.add_evidence(results)

    # 3. Also get related evidence from vector store
    related = await state.search_related(query, n_results=3)
    if related:
        await state.add_evidence(related)

    # 4. Return formatted string for LLM
    total_new = len(unique)
    total_stored = len(state.evidence_store)

    output = _format_results(results, "PubMed", query)
    output += f"\n[State: {total_new} new, {total_stored} total in evidence store]"

    if related:
        output += f"\n[Also found {len(related)} semantically related items from previous searches]"

    return output


@AIFunction
async def search_clinical_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for clinical studies.

    Use this tool to find ongoing and completed clinical trials
    for drug repurposing candidates.

    Args:
        query: Search terms (e.g., "metformin cancer phase 3")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of clinical trials with status and details
    """
    # 1. Execute search
    results = await _clinicaltrials.search(query, max_results)

    # 2. Update shared state
    state = get_magentic_state()
    unique = await state.add_evidence(results)

    # 3. Return formatted string
    total_new = len(unique)
    total_stored = len(state.evidence_store)

    output = _format_results(results, "ClinicalTrials.gov", query)
    output += f"\n[State: {total_new} new, {total_stored} total in evidence store]"

    return output


@AIFunction
async def search_preprints(query: str, max_results: int = 10) -> str:
    """Search bioRxiv/medRxiv for preprint papers.

    Use this tool to find the latest research that hasn't been
    peer-reviewed yet. Good for cutting-edge findings.

    Args:
        query: Search terms (e.g., "long covid treatment")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of preprints with abstracts and links
    """
    # 1. Execute search
    results = await _biorxiv.search(query, max_results)

    # 2. Update shared state
    state = get_magentic_state()
    unique = await state.add_evidence(results)

    # 3. Return formatted string
    total_new = len(unique)
    total_stored = len(state.evidence_store)

    output = _format_results(results, "bioRxiv/medRxiv", query)
    output += f"\n[State: {total_new} new, {total_stored} total in evidence store]"

    return output


@AIFunction
async def get_evidence_summary() -> str:
    """Get summary of all collected evidence.

    Use this tool when you need to review what evidence has been collected
    before making an assessment or generating a report.

    Returns:
        Summary of evidence store with counts and key citations
    """
    state = get_magentic_state()
    evidence = state.evidence_store

    if not evidence:
        return "No evidence collected yet."

    # Group by source
    by_source: dict[str, list] = {}
    for e in evidence:
        src = e.citation.source
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(e)

    output = [f"**Evidence Store Summary** ({len(evidence)} total items)\n"]

    for source, items in by_source.items():
        output.append(f"\n### {source.upper()} ({len(items)} items)")
        for e in items[:5]:  # First 5 per source
            output.append(f"- {e.citation.title[:80]}...")

    return "\n".join(output)


@AIFunction
async def get_bibliography() -> str:
    """Get full bibliography of all collected evidence.

    Use this tool when generating a final report to get properly
    formatted citations for all evidence.

    Returns:
        Numbered bibliography with full citation details
    """
    state = get_magentic_state()
    evidence = state.evidence_store

    if not evidence:
        return "No evidence collected for bibliography."

    output = ["## References\n"]

    for i, e in enumerate(evidence, 1):
        # Format: Authors (Year). Title. Source. URL
        authors = ", ".join(e.citation.authors[:3]) if e.citation.authors else "Unknown"
        if e.citation.authors and len(e.citation.authors) > 3:
            authors += " et al."

        year = e.citation.date[:4] if e.citation.date else "n.d."

        output.append(
            f"{i}. {authors} ({year}). {e.citation.title}. "
            f"*{e.citation.source.upper()}*. [{e.citation.url}]({e.citation.url})"
        )

    return "\n".join(output)
```

### 3.3 ChatAgent-Based Agents (`src/agents/magentic_agents.py`)

```python
"""Magentic-compatible agents using ChatAgent pattern."""
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from src.agents.tools import (
    get_bibliography,
    get_evidence_summary,
    search_clinical_trials,
    search_preprints,
    search_pubmed,
)
from src.utils.config import settings


def create_search_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a search agent with internal LLM and search tools.

    Args:
        chat_client: Optional custom chat client. If None, uses default.

    Returns:
        ChatAgent configured for biomedical search
    """
    client = chat_client or OpenAIChatClient(
        model_id="gpt-4o-mini",  # Fast, cheap for tool orchestration
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="SearchAgent",
        description="Searches biomedical databases (PubMed, ClinicalTrials.gov, bioRxiv) for drug repurposing evidence",
        instructions="""You are a biomedical search specialist. When asked to find evidence:

1. Analyze the request to determine what to search for
2. Extract key search terms (drug names, disease names, mechanisms)
3. Use the appropriate search tools:
   - search_pubmed for peer-reviewed papers
   - search_clinical_trials for clinical studies
   - search_preprints for cutting-edge findings
4. Summarize what you found and highlight key evidence

Be thorough - search multiple databases when appropriate.
Focus on finding: mechanisms of action, clinical evidence, and specific drug candidates.""",
        chat_client=client,
        tools=[search_pubmed, search_clinical_trials, search_preprints],
        temperature=0.3,  # More deterministic for tool use
    )


def create_judge_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a judge agent that evaluates evidence quality.

    Args:
        chat_client: Optional custom chat client. If None, uses default.

    Returns:
        ChatAgent configured for evidence assessment
    """
    client = chat_client or OpenAIChatClient(
        model_id="gpt-4o",  # Better model for nuanced judgment
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="JudgeAgent",
        description="Evaluates evidence quality and determines if sufficient for synthesis",
        instructions="""You are an evidence quality assessor. When asked to evaluate:

1. First, call get_evidence_summary() to see all collected evidence
2. Score on two dimensions (0-10 each):
   - Mechanism Score: How well is the biological mechanism explained?
   - Clinical Score: How strong is the clinical/preclinical evidence?
3. Determine if evidence is SUFFICIENT for a final report:
   - Sufficient: Clear mechanism + supporting clinical data
   - Insufficient: Gaps in mechanism OR weak clinical evidence
4. If insufficient, suggest specific search queries to fill gaps

Be rigorous but fair. Look for:
- Molecular targets and pathways
- Animal model studies
- Human clinical trials
- Safety data
- Drug-drug interactions""",
        chat_client=client,
        tools=[get_evidence_summary],  # Can review collected evidence
        temperature=0.2,  # Consistent judgments
    )


def create_hypothesis_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a hypothesis generation agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.

    Returns:
        ChatAgent configured for hypothesis generation
    """
    client = chat_client or OpenAIChatClient(
        model_id="gpt-4o",
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="HypothesisAgent",
        description="Generates mechanistic hypotheses for drug repurposing",
        instructions="""You are a biomedical hypothesis generator. Based on evidence:

1. Identify the key molecular targets involved
2. Map the biological pathways affected
3. Generate testable hypotheses in this format:

   DRUG → TARGET → PATHWAY → THERAPEUTIC EFFECT

   Example:
   Metformin → AMPK activation → mTOR inhibition → Reduced tau phosphorylation

4. Explain the rationale for each hypothesis
5. Suggest what additional evidence would support or refute it

Focus on mechanistic plausibility and existing evidence.""",
        chat_client=client,
        temperature=0.5,  # Some creativity for hypothesis generation
    )


def create_report_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a report synthesis agent.

    Args:
        chat_client: Optional custom chat client. If None, uses default.

    Returns:
        ChatAgent configured for report generation
    """
    client = chat_client or OpenAIChatClient(
        model_id="gpt-4o",
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="ReportAgent",
        description="Synthesizes research findings into structured reports",
        instructions="""You are a scientific report writer. When asked to synthesize:

1. First, call get_evidence_summary() to review all collected evidence
2. Then call get_bibliography() to get properly formatted citations

Generate a structured report with these sections:

## Executive Summary
Brief overview of findings and recommendation

## Methodology
Databases searched, queries used, evidence reviewed

## Key Findings
### Mechanism of Action
- Molecular targets
- Biological pathways
- Proposed mechanism

### Clinical Evidence
- Preclinical studies
- Clinical trials
- Safety profile

## Drug Candidates
List specific drugs with repurposing potential

## Limitations
Gaps in evidence, conflicting data, caveats

## Conclusion
Final recommendation with confidence level

## References
Use the output from get_bibliography() - do not make up citations!

Be comprehensive but concise. Cite evidence for all claims.""",
        chat_client=client,
        tools=[get_evidence_summary, get_bibliography],  # Access to collected evidence
        temperature=0.3,
    )
```

### 3.4 Magentic Orchestrator (`src/orchestrator_magentic.py`)

```python
"""Magentic-based orchestrator using ChatAgent pattern."""
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from agent_framework import (
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticOrchestratorMessageEvent,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

from src.agents.magentic_agents import (
    create_hypothesis_agent,
    create_judge_agent,
    create_report_agent,
    create_search_agent,
)
from src.agents.state import get_magentic_state, reset_magentic_state
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError
from src.utils.models import AgentEvent

logger = structlog.get_logger()


class MagenticOrchestrator:
    """
    Magentic-based orchestrator using ChatAgent pattern.

    Each agent has an internal LLM that understands natural language
    instructions from the manager and can call tools appropriately.
    """

    def __init__(
        self,
        max_rounds: int = 10,
        chat_client: OpenAIChatClient | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            max_rounds: Maximum coordination rounds
            chat_client: Optional shared chat client for agents
        """
        if not settings.openai_api_key:
            raise ConfigurationError(
                "Magentic mode requires OPENAI_API_KEY. "
                "Set the key or use mode='simple'."
            )

        self._max_rounds = max_rounds
        self._chat_client = chat_client

    def _build_workflow(self) -> Any:
        """Build the Magentic workflow with ChatAgent participants."""
        # Create agents with internal LLMs
        search_agent = create_search_agent(self._chat_client)
        judge_agent = create_judge_agent(self._chat_client)
        hypothesis_agent = create_hypothesis_agent(self._chat_client)
        report_agent = create_report_agent(self._chat_client)

        # Manager chat client (orchestrates the agents)
        manager_client = OpenAIChatClient(
            model_id="gpt-4o",  # Good model for planning/coordination
            api_key=settings.openai_api_key,
        )

        return (
            MagenticBuilder()
            .participants(
                searcher=search_agent,
                hypothesizer=hypothesis_agent,
                judge=judge_agent,
                reporter=report_agent,
            )
            .with_standard_manager(
                chat_client=manager_client,
                max_round_count=self._max_rounds,
                max_stall_count=3,
                max_reset_count=2,
            )
            .build()
        )

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the Magentic workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        logger.info("Starting Magentic orchestrator", query=query)

        # CRITICAL: Reset state for fresh workflow run
        reset_magentic_state()

        # Initialize embedding service if available
        state = get_magentic_state()
        state.init_embedding_service()

        yield AgentEvent(
            type="started",
            message=f"Starting research (Magentic mode): {query}",
            iteration=0,
        )

        workflow = self._build_workflow()

        task = f"""Research drug repurposing opportunities for: {query}

Workflow:
1. SearchAgent: Find evidence from PubMed, ClinicalTrials.gov, and bioRxiv
2. HypothesisAgent: Generate mechanistic hypotheses (Drug → Target → Pathway → Effect)
3. JudgeAgent: Evaluate if evidence is sufficient
4. If insufficient → SearchAgent refines search based on gaps
5. If sufficient → ReportAgent synthesizes final report

Focus on:
- Identifying specific molecular targets
- Understanding mechanism of action
- Finding clinical evidence supporting hypotheses

The final output should be a structured research report."""

        iteration = 0
        try:
            async for event in workflow.run_stream(task):
                agent_event = self._process_event(event, iteration)
                if agent_event:
                    if isinstance(event, MagenticAgentMessageEvent):
                        iteration += 1
                    yield agent_event

        except Exception as e:
            logger.error("Magentic workflow failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Workflow error: {e!s}",
                iteration=iteration,
            )

    def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
        """Process workflow event into AgentEvent."""
        if isinstance(event, MagenticOrchestratorMessageEvent):
            text = event.message.text if event.message else ""
            if text:
                return AgentEvent(
                    type="judging",
                    message=f"Manager ({event.kind}): {text[:200]}...",
                    iteration=iteration,
                )

        elif isinstance(event, MagenticAgentMessageEvent):
            agent_name = event.agent_id or "unknown"
            text = event.message.text if event.message else ""

            event_type = "judging"
            if "search" in agent_name.lower():
                event_type = "search_complete"
            elif "judge" in agent_name.lower():
                event_type = "judge_complete"
            elif "hypothes" in agent_name.lower():
                event_type = "hypothesizing"
            elif "report" in agent_name.lower():
                event_type = "synthesizing"

            return AgentEvent(
                type=event_type,
                message=f"{agent_name}: {text[:200]}...",
                iteration=iteration + 1,
            )

        elif isinstance(event, MagenticFinalResultEvent):
            text = event.message.text if event.message else "No result"
            return AgentEvent(
                type="complete",
                message=text,
                data={"iterations": iteration},
                iteration=iteration,
            )

        elif isinstance(event, MagenticAgentDeltaEvent):
            if event.text:
                return AgentEvent(
                    type="streaming",
                    message=event.text,
                    data={"agent_id": event.agent_id},
                    iteration=iteration,
                )

        elif isinstance(event, WorkflowOutputEvent):
            if event.data:
                return AgentEvent(
                    type="complete",
                    message=str(event.data),
                    iteration=iteration,
                )

        return None
```

### 3.4 Updated Factory (`src/orchestrator_factory.py`)

```python
"""Factory for creating orchestrators."""
from typing import Any, Literal

from src.orchestrator import JudgeHandlerProtocol, Orchestrator, SearchHandlerProtocol
from src.utils.models import OrchestratorConfig


def create_orchestrator(
    search_handler: SearchHandlerProtocol | None = None,
    judge_handler: JudgeHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic"] = "simple",
) -> Any:
    """
    Create an orchestrator instance.

    Args:
        search_handler: The search handler (required for simple mode)
        judge_handler: The judge handler (required for simple mode)
        config: Optional configuration
        mode: "simple" for Phase 4 loop, "magentic" for ChatAgent-based multi-agent

    Returns:
        Orchestrator instance

    Note:
        Magentic mode does NOT use search_handler/judge_handler.
        It creates ChatAgent instances with internal LLMs that call tools directly.
    """
    if mode == "magentic":
        try:
            from src.orchestrator_magentic import MagenticOrchestrator

            return MagenticOrchestrator(
                max_rounds=config.max_iterations if config else 10,
            )
        except ImportError:
            # Fallback to simple if agent-framework not installed
            pass

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

## 4. Why This Works

### 4.1 The Manager → Agent Communication

```
Manager LLM decides: "Tell SearchAgent to find clinical trials for metformin"
           ↓
Sends instruction: "Search for clinical trials about metformin and cancer"
           ↓
SearchAgent's INTERNAL LLM receives this
           ↓
Internal LLM understands: "I should call search_clinical_trials('metformin cancer')"
           ↓
Tool executes: ClinicalTrials.gov API
           ↓
Internal LLM formats response: "I found 15 trials. Here are the key ones..."
           ↓
Manager receives natural language response
```

### 4.2 Why Our Old Implementation Failed

```
Manager sends: "Search for clinical trials about metformin..."
           ↓
OLD SearchAgent.run() extracts: query = "Search for clinical trials about metformin..."
           ↓
Passes to PubMed: pubmed.search("Search for clinical trials about metformin...")
           ↓
PubMed doesn't understand English instructions → garbage results or error
```

---

## 5. Directory Structure

```text
src/
├── agents/
│   ├── __init__.py
│   ├── state.py                 # MagenticState (evidence_store + embeddings)
│   ├── tools.py                 # AIFunction tool definitions (update state)
│   └── magentic_agents.py       # ChatAgent factory functions
├── services/
│   └── embeddings.py            # EmbeddingService (semantic dedup)
├── orchestrator.py              # Simple mode (unchanged)
├── orchestrator_magentic.py     # Magentic mode with ChatAgents
└── orchestrator_factory.py      # Mode selection
```

---

## 6. Dependencies

```toml
[project.optional-dependencies]
magentic = [
    "agent-framework-core>=1.0.0b",
    "agent-framework-openai>=1.0.0b",  # For OpenAIChatClient
]
embeddings = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
]
```

**IMPORTANT: Magentic mode REQUIRES OpenAI API key.**

The Microsoft Agent Framework's standard manager and ChatAgent use OpenAIChatClient internally.
There is no AnthropicChatClient in the framework. If only `ANTHROPIC_API_KEY` is set:
- `mode="simple"` works fine
- `mode="magentic"` throws `ConfigurationError`

This is enforced in `MagenticOrchestrator.__init__`.

---

## 7. Implementation Checklist

- [ ] Create `src/agents/state.py` with MagenticState class
- [ ] Create `src/agents/tools.py` with AIFunction search tools + state updates
- [ ] Create `src/agents/magentic_agents.py` with ChatAgent factories
- [ ] Rewrite `src/orchestrator_magentic.py` to use ChatAgent pattern
- [ ] Update `src/orchestrator_factory.py` for new signature
- [ ] Test with real OpenAI API
- [ ] Verify manager properly coordinates agents
- [ ] Ensure tools are called with correct parameters
- [ ] Verify semantic deduplication works (evidence_store populates)
- [ ] Verify bibliography generation in final reports

---

## 8. Definition of Done

Phase 5 is **COMPLETE** when:

1. Magentic mode runs without hanging
2. Manager successfully coordinates agents via natural language
3. SearchAgent calls tools with proper search keywords (not raw instructions)
4. JudgeAgent evaluates evidence from conversation history
5. ReportAgent generates structured final report
6. Events stream to UI correctly

---

## 9. Testing Magentic Mode

```bash
# Test with real API
OPENAI_API_KEY=sk-... uv run python -c "
import asyncio
from src.orchestrator_factory import create_orchestrator

async def test():
    orch = create_orchestrator(mode='magentic')
    async for event in orch.run('metformin alzheimer'):
        print(f'[{event.type}] {event.message[:100]}')

asyncio.run(test())
"
```

Expected output:
```
[started] Starting research (Magentic mode): metformin alzheimer
[judging] Manager (plan): I will coordinate the agents to research...
[search_complete] SearchAgent: Found 25 PubMed results for metformin alzheimer...
[hypothesizing] HypothesisAgent: Based on the evidence, I propose...
[judge_complete] JudgeAgent: Mechanism Score: 7/10, Clinical Score: 6/10...
[synthesizing] ReportAgent: ## Executive Summary...
[complete] <full research report>
```

---

## 10. Key Differences from Old Spec

| Aspect | OLD (Wrong) | NEW (Correct) |
|--------|-------------|---------------|
| Agent type | `BaseAgent` subclass | `ChatAgent` with `chat_client` |
| Internal LLM | None | OpenAIChatClient |
| How tools work | Handler.execute(raw_instruction) | LLM understands instruction, calls AIFunction |
| Message handling | Extract text → pass to API | LLM interprets → extracts keywords → calls tool |
| State management | Passed to agent constructors | Global MagenticState singleton |
| Evidence storage | In agent instance | In MagenticState.evidence_store |
| Semantic search | Coupled to agents | Tools call state.add_evidence() |
| Citations for report | From agent's store | Via get_bibliography() tool |

**Key Insights:**
1. Magentic agents must have internal LLMs to understand natural language instructions
2. Tools must update shared state as a side effect (return strings, but also store Evidence)
3. ReportAgent uses `get_bibliography()` tool to access structured citations
4. State is reset at start of each workflow run via `reset_magentic_state()`
