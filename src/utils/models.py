"""Data models for the Search feature."""

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

# Centralized source type - add new sources here (e.g., "biorxiv" in Phase 11)
SourceName = Literal["pubmed", "clinicaltrials", "biorxiv", "europepmc", "preprint", "rag", "web"]


class Citation(BaseModel):
    """A citation to a source document."""

    source: SourceName = Field(description="Where this came from")

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(description="URL to the source")
    date: str = Field(description="Publication date (YYYY-MM-DD or 'Unknown')")
    authors: list[str] = Field(default_factory=list)

    MAX_AUTHORS_IN_CITATION: ClassVar[int] = 3

    @property
    def formatted(self) -> str:
        """Format as a citation string."""
        author_str = ", ".join(self.authors[: self.MAX_AUTHORS_IN_CITATION])
        if len(self.authors) > self.MAX_AUTHORS_IN_CITATION:
            author_str += " et al."
        return f"{author_str} ({self.date}). {self.title}. {self.source.upper()}"


class Evidence(BaseModel):
    """A piece of evidence retrieved from search."""

    content: str = Field(min_length=1, description="The actual text content")
    citation: Citation
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score 0-1")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., cited_by_count, concepts, is_open_access)",
    )

    model_config = {"frozen": True}


class SearchResult(BaseModel):
    """Result of a search operation."""

    query: str
    evidence: list[Evidence]
    sources_searched: list[SourceName]
    total_found: int
    errors: list[str] = Field(default_factory=list)


class AssessmentDetails(BaseModel):
    """Detailed assessment of evidence quality."""

    mechanism_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How well does the evidence explain the mechanism? 0-10",
    )
    mechanism_reasoning: str = Field(
        ..., min_length=10, description="Explanation of mechanism score"
    )
    clinical_evidence_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Strength of clinical/preclinical evidence. 0-10",
    )
    clinical_reasoning: str = Field(
        ..., min_length=10, description="Explanation of clinical evidence score"
    )
    drug_candidates: list[str] = Field(
        default_factory=list, description="List of specific drug candidates mentioned"
    )
    key_findings: list[str] = Field(
        default_factory=list, description="Key findings from the evidence"
    )


class JudgeAssessment(BaseModel):
    """Complete assessment from the Judge."""

    details: AssessmentDetails
    sufficient: bool = Field(..., description="Is evidence sufficient to provide a recommendation?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment (0-1)")
    recommendation: Literal["continue", "synthesize"] = Field(
        ...,
        description="continue = need more evidence, synthesize = ready to answer",
    )
    next_search_queries: list[str] = Field(
        default_factory=list, description="If continue, what queries to search next"
    )
    reasoning: str = Field(
        ..., min_length=20, description="Overall reasoning for the recommendation"
    )


class AgentEvent(BaseModel):
    """Event emitted by the orchestrator for UI streaming."""

    type: Literal[
        "started",
        "searching",
        "search_complete",
        "judging",
        "judge_complete",
        "looping",
        "synthesizing",
        "complete",
        "error",
        "streaming",
        "hypothesizing",
        "analyzing",  # NEW for Phase 13
        "analysis_complete",  # NEW for Phase 13
    ]
    message: str
    data: Any = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    iteration: int = 0

    def to_markdown(self) -> str:
        """Format event as markdown for chat display."""
        icons = {
            "started": "🚀",
            "searching": "🔍",
            "search_complete": "📚",
            "judging": "🧠",
            "judge_complete": "✅",
            "looping": "🔄",
            "synthesizing": "📝",
            "complete": "🎉",
            "error": "❌",
            "streaming": "📡",
            "hypothesizing": "🔬",  # NEW
            "analyzing": "📊",  # NEW
            "analysis_complete": "📈",  # NEW
        }
        icon = icons.get(self.type, "•")
        return f"{icon} **{self.type.upper()}**: {self.message}"


class MechanismHypothesis(BaseModel):
    """A scientific hypothesis about drug mechanism."""

    drug: str = Field(description="The drug being studied")
    target: str = Field(description="Molecular target (e.g., AMPK, mTOR)")
    pathway: str = Field(description="Biological pathway affected")
    effect: str = Field(description="Downstream effect on disease")
    confidence: float = Field(ge=0, le=1, description="Confidence in hypothesis")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="PMIDs or URLs supporting this hypothesis"
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list, description="PMIDs or URLs contradicting this hypothesis"
    )
    search_suggestions: list[str] = Field(
        default_factory=list, description="Suggested searches to test this hypothesis"
    )

    def to_search_queries(self) -> list[str]:
        """Generate search queries to test this hypothesis."""
        return [
            f"{self.drug} {self.target}",
            f"{self.target} {self.pathway}",
            f"{self.pathway} {self.effect}",
            *self.search_suggestions,
        ]


class HypothesisAssessment(BaseModel):
    """Assessment of evidence against hypotheses."""

    hypotheses: list[MechanismHypothesis]
    primary_hypothesis: MechanismHypothesis | None = Field(
        default=None, description="Most promising hypothesis based on current evidence"
    )
    knowledge_gaps: list[str] = Field(description="What we don't know yet")
    recommended_searches: list[str] = Field(description="Searches to fill knowledge gaps")


class ReportSection(BaseModel):
    """A section of the research report."""

    title: str
    content: str
    # Reserved for future inline citation tracking within sections
    citations: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Structured scientific report."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(
        description="One-paragraph summary for quick reading", min_length=100, max_length=1000
    )
    research_question: str = Field(description="Clear statement of what was investigated")

    methodology: ReportSection = Field(description="How the research was conducted")
    hypotheses_tested: list[dict[str, Any]] = Field(
        description="Hypotheses with supporting/contradicting evidence counts"
    )

    mechanistic_findings: ReportSection = Field(description="Findings about drug mechanisms")
    clinical_findings: ReportSection = Field(
        description="Findings from clinical/preclinical studies"
    )

    drug_candidates: list[str] = Field(description="Identified drug candidates")
    limitations: list[str] = Field(description="Study limitations")
    conclusion: str = Field(description="Overall conclusion")

    references: list[dict[str, str]] = Field(
        default_factory=list,
        description="Formatted references with title, authors, source, URL",
    )

    # Metadata
    sources_searched: list[str] = Field(default_factory=list)
    total_papers_reviewed: int = 0
    search_iterations: int = 0
    confidence_score: float = Field(ge=0, le=1)

    def to_markdown(self) -> str:
        """Render report as markdown."""
        sections = [
            f"# {self.title}\n",
            f"## Executive Summary\n{self.executive_summary}\n",
            f"## Research Question\n{self.research_question}\n",
            f"## Methodology\n{self.methodology.content}\n",
        ]

        # Hypotheses
        sections.append("## Hypotheses Tested\n")
        if not self.hypotheses_tested:
            sections.append("*No hypotheses tested yet.*\n")
        for h in self.hypotheses_tested:
            supported = h.get("supported", 0)
            contradicted = h.get("contradicted", 0)
            if supported == 0 and contradicted == 0:
                status = "❓ Untested"
            elif supported > contradicted:
                status = "✅ Supported"
            else:
                status = "⚠️ Mixed"
            sections.append(
                f"- **{h.get('mechanism', 'Unknown')}** ({status}): "
                f"{supported} supporting, {contradicted} contradicting\n"
            )

        # Findings
        sections.append(f"## Mechanistic Findings\n{self.mechanistic_findings.content}\n")
        sections.append(f"## Clinical Findings\n{self.clinical_findings.content}\n")

        # Drug candidates
        sections.append("## Drug Candidates\n")
        if self.drug_candidates:
            for drug in self.drug_candidates:
                sections.append(f"- **{drug}**\n")
        else:
            sections.append("*No drug candidates identified.*\n")

        # Limitations
        sections.append("## Limitations\n")
        if self.limitations:
            for lim in self.limitations:
                sections.append(f"- {lim}\n")
        else:
            sections.append("*No limitations documented.*\n")

        # Conclusion
        sections.append(f"## Conclusion\n{self.conclusion}\n")

        # References
        sections.append("## References\n")
        if self.references:
            for i, ref in enumerate(self.references, 1):
                sections.append(
                    f"{i}. {ref.get('authors', 'Unknown')}. "
                    f"*{ref.get('title', 'Untitled')}*. "
                    f"{ref.get('source', '')} ({ref.get('date', '')}). "
                    f"[Link]({ref.get('url', '#')})\n"
                )
        else:
            sections.append("*No references available.*\n")

        # Metadata footer
        sections.append("\n---\n")
        sections.append(
            f"*Report generated from {self.total_papers_reviewed} papers "
            f"across {self.search_iterations} search iterations. "
            f"Confidence: {self.confidence_score:.0%}*"
        )

        return "\n".join(sections)


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_iterations: int = Field(default=10, ge=1, le=20)
    max_results_per_tool: int = Field(default=10, ge=1, le=50)
    search_timeout: float = Field(default=30.0, ge=5.0, le=120.0)


# Models for iterative/deep research patterns


class IterationData(BaseModel):
    """Data for a single iteration of the research loop."""

    gap: str = Field(description="The gap addressed in the iteration", default="")
    tool_calls: list[str] = Field(description="The tool calls made", default_factory=list)
    findings: list[str] = Field(
        description="The findings collected from tool calls", default_factory=list
    )
    thought: str = Field(
        description="The thinking done to reflect on the success of the iteration and next steps",
        default="",
    )

    model_config = {"frozen": True}


class Conversation(BaseModel):
    """A conversation between the user and the iterative researcher."""

    history: list[IterationData] = Field(
        description="The data for each iteration of the research loop",
        default_factory=list,
    )

    def add_iteration(self, iteration_data: IterationData | None = None) -> None:
        """Add a new iteration to the conversation history."""
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)

    def set_latest_gap(self, gap: str) -> None:
        """Set the gap for the latest iteration."""
        if not self.history:
            self.add_iteration()
        # Use model_copy() since IterationData is frozen
        self.history[-1] = self.history[-1].model_copy(update={"gap": gap})

    def set_latest_tool_calls(self, tool_calls: list[str]) -> None:
        """Set the tool calls for the latest iteration."""
        if not self.history:
            self.add_iteration()
        # Use model_copy() since IterationData is frozen
        self.history[-1] = self.history[-1].model_copy(update={"tool_calls": tool_calls})

    def set_latest_findings(self, findings: list[str]) -> None:
        """Set the findings for the latest iteration."""
        if not self.history:
            self.add_iteration()
        # Use model_copy() since IterationData is frozen
        self.history[-1] = self.history[-1].model_copy(update={"findings": findings})

    def set_latest_thought(self, thought: str) -> None:
        """Set the thought for the latest iteration."""
        if not self.history:
            self.add_iteration()
        # Use model_copy() since IterationData is frozen
        self.history[-1] = self.history[-1].model_copy(update={"thought": thought})

    def get_latest_gap(self) -> str:
        """Get the gap from the latest iteration."""
        if not self.history:
            return ""
        return self.history[-1].gap

    def get_latest_tool_calls(self) -> list[str]:
        """Get the tool calls from the latest iteration."""
        if not self.history:
            return []
        return self.history[-1].tool_calls

    def get_latest_findings(self) -> list[str]:
        """Get the findings from the latest iteration."""
        if not self.history:
            return []
        return self.history[-1].findings

    def get_latest_thought(self) -> str:
        """Get the thought from the latest iteration."""
        if not self.history:
            return ""
        return self.history[-1].thought

    def get_all_findings(self) -> list[str]:
        """Get all findings from all iterations."""
        return [finding for iteration_data in self.history for finding in iteration_data.findings]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""
        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"{self.get_thought_string(iteration_num)}\n\n"
            if iteration_data.gap:
                conversation += f"{self.get_task_string(iteration_num)}\n\n"
            if iteration_data.tool_calls:
                conversation += f"{self.get_action_string(iteration_num)}\n\n"
            if iteration_data.findings:
                conversation += f"{self.get_findings_string(iteration_num)}\n\n"

        return conversation

    def get_task_string(self, iteration_num: int) -> str:
        """Get the task for the specified iteration."""
        if iteration_num < len(self.history) and self.history[iteration_num].gap:
            return (
                f"<task>\nAddress this knowledge gap: "
                f"{self.history[iteration_num].gap}\n</task>"
            )
        return ""

    def get_action_string(self, iteration_num: int) -> str:
        """Get the action for the specified iteration."""
        if iteration_num < len(self.history) and self.history[iteration_num].tool_calls:
            joined_calls = "\n".join(self.history[iteration_num].tool_calls)
            return (
                "<action>\nCalling the following tools to address the knowledge gap:\n"
                f"{joined_calls}\n</action>"
            )
        return ""

    def get_findings_string(self, iteration_num: int) -> str:
        """Get the findings for the specified iteration."""
        if iteration_num < len(self.history) and self.history[iteration_num].findings:
            joined_findings = "\n\n".join(self.history[iteration_num].findings)
            return f"<findings>\n{joined_findings}\n</findings>"
        return ""

    def get_thought_string(self, iteration_num: int) -> str:
        """Get the thought for the specified iteration."""
        if iteration_num < len(self.history) and self.history[iteration_num].thought:
            return f"<thought>\n{self.history[iteration_num].thought}\n</thought>"
        return ""

    def latest_task_string(self) -> str:
        """Get the latest task."""
        if not self.history:
            return ""
        return self.get_task_string(len(self.history) - 1)

    def latest_action_string(self) -> str:
        """Get the latest action."""
        if not self.history:
            return ""
        return self.get_action_string(len(self.history) - 1)

    def latest_findings_string(self) -> str:
        """Get the latest findings."""
        if not self.history:
            return ""
        return self.get_findings_string(len(self.history) - 1)

    def latest_thought_string(self) -> str:
        """Get the latest thought."""
        if not self.history:
            return ""
        return self.get_thought_string(len(self.history) - 1)


class ReportPlanSection(BaseModel):
    """A section of the report that needs to be written."""

    title: str = Field(description="The title of the section")
    key_question: str = Field(description="The key question to be addressed in the section")

    model_config = {"frozen": True}


class ReportPlan(BaseModel):
    """Output from the Report Planner Agent."""

    background_context: str = Field(
        description="A summary of supporting context that can be passed onto the research agents"
    )
    report_outline: list[ReportPlanSection] = Field(
        description="List of sections that need to be written in the report"
    )
    report_title: str = Field(description="The title of the report")

    model_config = {"frozen": True}


class KnowledgeGapOutput(BaseModel):
    """Output from the Knowledge Gap Agent."""

    research_complete: bool = Field(
        description="Whether the research and findings are complete enough to end the research loop"
    )
    outstanding_gaps: list[str] = Field(
        description="List of knowledge gaps that still need to be addressed"
    )

    model_config = {"frozen": True}


class AgentTask(BaseModel):
    """A task for a specific agent to address knowledge gaps."""

    gap: str | None = Field(description="The knowledge gap being addressed", default=None)
    agent: str = Field(description="The name of the agent to use")
    query: str = Field(description="The specific query for the agent")
    entity_website: str | None = Field(
        description="The website of the entity being researched, if known",
        default=None,
    )

    model_config = {"frozen": True}


class AgentSelectionPlan(BaseModel):
    """Plan for which agents to use for knowledge gaps."""

    tasks: list[AgentTask] = Field(description="List of agent tasks to address knowledge gaps")

    model_config = {"frozen": True}


class ReportDraftSection(BaseModel):
    """A section of the report that needs to be written."""

    section_title: str = Field(description="The title of the section")
    section_content: str = Field(description="The content of the section")

    model_config = {"frozen": True}


class ReportDraft(BaseModel):
    """Output from the Report Planner Agent."""

    sections: list[ReportDraftSection] = Field(
        description="List of sections that are in the report"
    )

    model_config = {"frozen": True}


class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents."""

    output: str = Field(description="The output from the tool agent")
    sources: list[str] = Field(description="List of source URLs", default_factory=list)

    model_config = {"frozen": True}


class ParsedQuery(BaseModel):
    """Parsed and improved user query with research mode detection."""

    original_query: str = Field(description="The original user query")
    improved_query: str = Field(description="Improved/refined query")
    research_mode: Literal["iterative", "deep"] = Field(description="Detected research mode")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Key entities extracted from query",
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="Specific research questions extracted",
    )

    model_config = {"frozen": True}
