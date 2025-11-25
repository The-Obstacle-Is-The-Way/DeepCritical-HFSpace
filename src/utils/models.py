"""Data models for the Search feature."""

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source document."""

    source: Literal["pubmed", "web"] = Field(description="Where this came from")
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
    sources_searched: list[Literal["pubmed", "web"]]
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
        "hypothesizing",  # NEW for Phase 7
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


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_iterations: int = Field(default=5, ge=1, le=10)
    max_results_per_tool: int = Field(default=10, ge=1, le=50)
    search_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
