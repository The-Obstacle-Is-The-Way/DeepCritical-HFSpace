"""Data models for the Search feature."""

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

# Centralized source type - add new sources here (e.g., "biorxiv" in Phase 11)
SourceName = Literal["pubmed", "clinicaltrials", "biorxiv"]


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
        description="Formatted references with title, authors, source, URL"
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

    max_iterations: int = Field(default=5, ge=1, le=10)
    max_results_per_tool: int = Field(default=10, ge=1, le=50)
    search_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
