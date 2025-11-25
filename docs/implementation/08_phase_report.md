# Phase 8 Implementation Spec: Report Agent

**Goal**: Generate structured scientific reports with proper citations and methodology.
**Philosophy**: "Research isn't complete until it's communicated clearly."
**Prerequisite**: Phase 7 complete (Hypothesis Agent working)

---

## 1. Why Report Agent?

Current limitation: **Synthesis is basic markdown, not a scientific report.**

Current output:
```markdown
## Drug Repurposing Analysis
### Drug Candidates
- Metformin
### Key Findings
- Some findings
### Citations
1. [Paper 1](url)
```

With Report Agent:
```markdown
## Executive Summary
One-paragraph summary for busy readers...

## Research Question
Clear statement of what was investigated...

## Methodology
- Sources searched: PubMed, DuckDuckGo
- Date range: ...
- Inclusion criteria: ...

## Hypotheses Tested
1. Metformin → AMPK → neuroprotection (Supported: 7 papers, Contradicted: 2)

## Findings
### Mechanistic Evidence
...
### Clinical Evidence
...

## Limitations
- Only English language papers
- Abstract-level analysis only

## Conclusion
...

## References
Properly formatted citations...
```

---

## 2. Architecture

### Phase 8 Addition
```text
Evidence + Hypotheses + Assessment
            ↓
      Report Agent
            ↓
   Structured Scientific Report
```

### Report Generation Flow
```text
1. JudgeAgent says "synthesize"
2. Magentic Manager selects ReportAgent
3. ReportAgent gathers:
   - All evidence from shared context
   - All hypotheses (supported/contradicted)
   - Assessment scores
4. ReportAgent generates structured report
5. Final output to user
```

---

## 3. Report Model

### 3.1 Data Model (`src/utils/models.py`)

```python
class ReportSection(BaseModel):
    """A section of the research report."""
    title: str
    content: str
    citations: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Structured scientific report."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(
        description="One-paragraph summary for quick reading",
        min_length=100,
        max_length=500
    )
    research_question: str = Field(description="Clear statement of what was investigated")

    methodology: ReportSection = Field(description="How the research was conducted")
    hypotheses_tested: list[dict] = Field(
        description="Hypotheses with supporting/contradicting evidence counts"
    )

    mechanistic_findings: ReportSection = Field(
        description="Findings about drug mechanisms"
    )
    clinical_findings: ReportSection = Field(
        description="Findings from clinical/preclinical studies"
    )

    drug_candidates: list[str] = Field(description="Identified drug candidates")
    limitations: list[str] = Field(description="Study limitations")
    conclusion: str = Field(description="Overall conclusion")

    references: list[dict] = Field(
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
        for h in self.hypotheses_tested:
            status = "✅ Supported" if h.get("supported", 0) > h.get("contradicted", 0) else "⚠️ Mixed"
            sections.append(
                f"- **{h['mechanism']}** ({status}): "
                f"{h.get('supported', 0)} supporting, {h.get('contradicted', 0)} contradicting\n"
            )

        # Findings
        sections.append(f"## Mechanistic Findings\n{self.mechanistic_findings.content}\n")
        sections.append(f"## Clinical Findings\n{self.clinical_findings.content}\n")

        # Drug candidates
        sections.append("## Drug Candidates\n")
        for drug in self.drug_candidates:
            sections.append(f"- **{drug}**\n")

        # Limitations
        sections.append("## Limitations\n")
        for lim in self.limitations:
            sections.append(f"- {lim}\n")

        # Conclusion
        sections.append(f"## Conclusion\n{self.conclusion}\n")

        # References
        sections.append("## References\n")
        for i, ref in enumerate(self.references, 1):
            sections.append(
                f"{i}. {ref.get('authors', 'Unknown')}. "
                f"*{ref.get('title', 'Untitled')}*. "
                f"{ref.get('source', '')} ({ref.get('date', '')}). "
                f"[Link]({ref.get('url', '#')})\n"
            )

        # Metadata footer
        sections.append("\n---\n")
        sections.append(
            f"*Report generated from {self.total_papers_reviewed} papers "
            f"across {self.search_iterations} search iterations. "
            f"Confidence: {self.confidence_score:.0%}*"
        )

        return "\n".join(sections)
```

---

## 4. Implementation

### 4.0 Citation Validation (`src/utils/citation_validator.py`)

> **🚨 CRITICAL: Why Citation Validation?**
>
> LLMs frequently **hallucinate** citations - inventing paper titles, authors, and URLs
> that don't exist. For a medical research tool, fake citations are **dangerous**.
>
> This validation layer ensures every reference in the report actually exists
> in the collected evidence.

```python
"""Citation validation to prevent LLM hallucination.

CRITICAL: Medical research requires accurate citations.
This module validates that all references exist in collected evidence.
"""
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.models import Evidence, ResearchReport

logger = logging.getLogger(__name__)


def validate_references(
    report: "ResearchReport",
    evidence: list["Evidence"]
) -> "ResearchReport":
    """Ensure all references actually exist in collected evidence.

    CRITICAL: Prevents LLM hallucination of citations.

    Args:
        report: The generated research report
        evidence: All evidence collected during research

    Returns:
        Report with only valid references (hallucinated ones removed)
    """
    # Build set of valid URLs from evidence
    valid_urls = {e.citation.url for e in evidence}
    valid_titles = {e.citation.title.lower() for e in evidence}

    validated_refs = []
    removed_count = 0

    for ref in report.references:
        ref_url = ref.get("url", "")
        ref_title = ref.get("title", "").lower()

        # Check if URL matches collected evidence
        if ref_url in valid_urls:
            validated_refs.append(ref)
        # Fallback: check title match (URLs might differ slightly)
        elif ref_title and any(ref_title in t or t in ref_title for t in valid_titles):
            validated_refs.append(ref)
        else:
            removed_count += 1
            logger.warning(
                f"Removed hallucinated reference: '{ref.get('title', 'Unknown')}' "
                f"(URL: {ref_url[:50]}...)"
            )

    if removed_count > 0:
        logger.info(
            f"Citation validation removed {removed_count} hallucinated references. "
            f"{len(validated_refs)} valid references remain."
        )

    # Update report with validated references
    report.references = validated_refs
    return report


def build_reference_from_evidence(evidence: "Evidence") -> dict:
    """Build a properly formatted reference from evidence.

    Use this to ensure references match the original evidence exactly.
    """
    return {
        "title": evidence.citation.title,
        "authors": evidence.citation.authors or ["Unknown"],
        "source": evidence.citation.source,
        "date": evidence.citation.date or "n.d.",
        "url": evidence.citation.url,
    }
```

### 4.1 Report Prompts (`src/prompts/report.py`)

```python
"""Prompts for Report Agent."""
from src.utils.text_utils import truncate_at_sentence, select_diverse_evidence

SYSTEM_PROMPT = """You are a scientific writer specializing in drug repurposing research reports.

Your role is to synthesize evidence and hypotheses into a clear, structured report.

A good report:
1. Has a clear EXECUTIVE SUMMARY (one paragraph, key takeaways)
2. States the RESEARCH QUESTION clearly
3. Describes METHODOLOGY (what was searched, how)
4. Evaluates HYPOTHESES with evidence counts
5. Separates MECHANISTIC and CLINICAL findings
6. Lists specific DRUG CANDIDATES
7. Acknowledges LIMITATIONS honestly
8. Provides a balanced CONCLUSION
9. Includes properly formatted REFERENCES

Write in scientific but accessible language. Be specific about evidence strength.

─────────────────────────────────────────────────────────────────────────────
🚨 CRITICAL CITATION REQUIREMENTS 🚨
─────────────────────────────────────────────────────────────────────────────

You MUST follow these rules for the References section:

1. You may ONLY cite papers that appear in the Evidence section above
2. Every reference URL must EXACTLY match a provided evidence URL
3. Do NOT invent, fabricate, or hallucinate any references
4. Do NOT modify paper titles, authors, dates, or URLs
5. If unsure about a citation, OMIT it rather than guess
6. Copy URLs exactly as provided - do not create similar-looking URLs

VIOLATION OF THESE RULES PRODUCES DANGEROUS MISINFORMATION.
─────────────────────────────────────────────────────────────────────────────"""


async def format_report_prompt(
    query: str,
    evidence: list,
    hypotheses: list,
    assessment: dict,
    metadata: dict,
    embeddings=None
) -> str:
    """Format prompt for report generation.

    Includes full evidence details for accurate citation.
    """
    # Select diverse evidence (not arbitrary truncation)
    selected = await select_diverse_evidence(
        evidence, n=20, query=query, embeddings=embeddings
    )

    # Include FULL citation details for each evidence item
    # This helps the LLM create accurate references
    evidence_summary = "\n".join([
        f"- **Title**: {e.citation.title}\n"
        f"  **URL**: {e.citation.url}\n"
        f"  **Authors**: {', '.join(e.citation.authors or ['Unknown'])}\n"
        f"  **Date**: {e.citation.date or 'n.d.'}\n"
        f"  **Source**: {e.citation.source}\n"
        f"  **Content**: {truncate_at_sentence(e.content, 200)}\n"
        for e in selected
    ])

    hypotheses_summary = "\n".join([
        f"- {h.drug} → {h.target} → {h.pathway} → {h.effect} (Confidence: {h.confidence:.0%})"
        for h in hypotheses
    ]) if hypotheses else "No hypotheses generated yet."

    return f"""Generate a structured research report for the following query.

## Original Query
{query}

## Evidence Collected ({len(selected)} papers, selected for diversity)

{evidence_summary}

## Hypotheses Generated
{hypotheses_summary}

## Assessment Scores
- Mechanism Score: {assessment.get('mechanism_score', 'N/A')}/10
- Clinical Evidence Score: {assessment.get('clinical_score', 'N/A')}/10
- Overall Confidence: {assessment.get('confidence', 0):.0%}

## Metadata
- Sources Searched: {', '.join(metadata.get('sources', []))}
- Search Iterations: {metadata.get('iterations', 0)}

Generate a complete ResearchReport with all sections filled in.

REMINDER: Only cite papers from the Evidence section above. Copy URLs exactly."""
```

### 4.2 Report Agent (`src/agents/report_agent.py`)

```python
"""Report agent for generating structured research reports."""
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)
from pydantic_ai import Agent

from src.prompts.report import SYSTEM_PROMPT, format_report_prompt
from src.utils.citation_validator import validate_references  # CRITICAL
from src.utils.config import settings
from src.utils.models import Evidence, MechanismHypothesis, ResearchReport

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class ReportAgent(BaseAgent):
    """Generates structured scientific reports from evidence and hypotheses."""

    def __init__(
        self,
        evidence_store: dict[str, list[Evidence]],
        embedding_service: "EmbeddingService | None" = None,  # For diverse selection
    ) -> None:
        super().__init__(
            name="ReportAgent",
            description="Generates structured scientific research reports with citations",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service
        self._agent = Agent(
            model=settings.llm_provider,
            output_type=ResearchReport,
            system_prompt=SYSTEM_PROMPT,
        )

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Generate research report."""
        query = self._extract_query(messages)

        # Gather all context
        evidence = self._evidence_store.get("current", [])
        hypotheses = self._evidence_store.get("hypotheses", [])
        assessment = self._evidence_store.get("last_assessment", {})

        if not evidence:
            return AgentRunResponse(
                messages=[ChatMessage(
                    role=Role.ASSISTANT,
                    text="Cannot generate report: No evidence collected."
                )],
                response_id="report-no-evidence",
            )

        # Build metadata
        metadata = {
            "sources": list(set(e.citation.source for e in evidence)),
            "iterations": self._evidence_store.get("iteration_count", 0),
        }

        # Generate report (format_report_prompt is now async)
        prompt = await format_report_prompt(
            query=query,
            evidence=evidence,
            hypotheses=hypotheses,
            assessment=assessment,
            metadata=metadata,
            embeddings=self._embeddings,
        )

        result = await self._agent.run(prompt)
        report = result.output

        # ═══════════════════════════════════════════════════════════════════
        # 🚨 CRITICAL: Validate citations to prevent hallucination
        # ═══════════════════════════════════════════════════════════════════
        report = validate_references(report, evidence)

        # Store validated report
        self._evidence_store["final_report"] = report

        # Return markdown version
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=report.to_markdown())],
            response_id="report-complete",
            additional_properties={"report": report.model_dump()},
        )

    def _extract_query(self, messages) -> str:
        """Extract query from messages."""
        if isinstance(messages, str):
            return messages
        elif isinstance(messages, ChatMessage):
            return messages.text or ""
        elif isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER:
                    return msg.text or ""
                elif isinstance(msg, str):
                    return msg
        return ""

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper."""
        result = await self.run(messages, thread=thread, **kwargs)
        yield AgentRunResponseUpdate(
            messages=result.messages,
            response_id=result.response_id
        )
```

### 4.3 Update MagenticOrchestrator

Add ReportAgent as the final synthesis step:

```python
# In MagenticOrchestrator.__init__
self._report_agent = ReportAgent(self._evidence_store)

# In workflow building
workflow = (
    MagenticBuilder()
    .participants(
        searcher=search_agent,
        hypothesizer=hypothesis_agent,
        judge=judge_agent,
        reporter=self._report_agent,  # NEW
    )
    .with_standard_manager(...)
    .build()
)

# Update task instruction
task = f"""Research drug repurposing opportunities for: {query}

Workflow:
1. SearchAgent: Find evidence from PubMed and web
2. HypothesisAgent: Generate mechanistic hypotheses
3. SearchAgent: Targeted search based on hypotheses
4. JudgeAgent: Evaluate evidence sufficiency
5. If sufficient → ReportAgent: Generate structured research report
6. If not sufficient → Repeat from step 1 with refined queries

The final output should be a complete research report with:
- Executive summary
- Methodology
- Hypotheses tested
- Mechanistic and clinical findings
- Drug candidates
- Limitations
- Conclusion with references
"""
```

---

## 5. Directory Structure After Phase 8

```
src/
├── agents/
│   ├── search_agent.py
│   ├── judge_agent.py
│   ├── hypothesis_agent.py
│   └── report_agent.py         # NEW
├── prompts/
│   ├── judge.py
│   ├── hypothesis.py
│   └── report.py               # NEW
├── services/
│   └── embeddings.py
└── utils/
    └── models.py               # Updated with report models
```

---

## 6. Tests

### 6.1 Unit Tests (`tests/unit/agents/test_report_agent.py`)

```python
"""Unit tests for ReportAgent."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.report_agent import ReportAgent
from src.utils.models import (
    Citation, Evidence, MechanismHypothesis,
    ResearchReport, ReportSection
)


@pytest.fixture
def sample_evidence():
    return [
        Evidence(
            content="Metformin activates AMPK...",
            citation=Citation(
                source="pubmed",
                title="Metformin mechanisms",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
                authors=["Smith J", "Jones A"]
            )
        )
    ]


@pytest.fixture
def sample_hypotheses():
    return [
        MechanismHypothesis(
            drug="Metformin",
            target="AMPK",
            pathway="mTOR inhibition",
            effect="Neuroprotection",
            confidence=0.8,
            search_suggestions=[]
        )
    ]


@pytest.fixture
def mock_report():
    return ResearchReport(
        title="Drug Repurposing Analysis: Metformin for Alzheimer's",
        executive_summary="This report analyzes metformin as a potential...",
        research_question="Can metformin be repurposed for Alzheimer's disease?",
        methodology=ReportSection(
            title="Methodology",
            content="Searched PubMed and web sources..."
        ),
        hypotheses_tested=[
            {"mechanism": "Metformin → AMPK → neuroprotection", "supported": 5, "contradicted": 1}
        ],
        mechanistic_findings=ReportSection(
            title="Mechanistic Findings",
            content="Evidence suggests AMPK activation..."
        ),
        clinical_findings=ReportSection(
            title="Clinical Findings",
            content="Limited clinical data available..."
        ),
        drug_candidates=["Metformin"],
        limitations=["Abstract-level analysis only"],
        conclusion="Metformin shows promise...",
        references=[],
        sources_searched=["pubmed", "web"],
        total_papers_reviewed=10,
        search_iterations=3,
        confidence_score=0.75
    )


@pytest.mark.asyncio
async def test_report_agent_generates_report(
    sample_evidence, sample_hypotheses, mock_report
):
    """ReportAgent should generate structured report."""
    store = {
        "current": sample_evidence,
        "hypotheses": sample_hypotheses,
        "last_assessment": {"mechanism_score": 8, "clinical_score": 6}
    }

    with patch("src.agents.report_agent.Agent") as MockAgent:
        mock_result = MagicMock()
        mock_result.output = mock_report
        MockAgent.return_value.run = AsyncMock(return_value=mock_result)

        agent = ReportAgent(store)
        response = await agent.run("metformin alzheimer")

        assert "Executive Summary" in response.messages[0].text
        assert "Methodology" in response.messages[0].text
        assert "References" in response.messages[0].text


@pytest.mark.asyncio
async def test_report_agent_no_evidence():
    """ReportAgent should handle empty evidence gracefully."""
    store = {"current": [], "hypotheses": []}
    agent = ReportAgent(store)

    response = await agent.run("test query")

    assert "Cannot generate report" in response.messages[0].text


# ═══════════════════════════════════════════════════════════════════════════
# 🚨 CRITICAL: Citation Validation Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_report_agent_removes_hallucinated_citations(sample_evidence):
    """ReportAgent should remove citations not in evidence."""
    from src.utils.citation_validator import validate_references

    # Create report with mix of valid and hallucinated references
    report_with_hallucinations = ResearchReport(
        title="Test Report",
        executive_summary="This is a test report for citation validation...",
        research_question="Testing citation validation",
        methodology=ReportSection(title="Methodology", content="Test"),
        hypotheses_tested=[],
        mechanistic_findings=ReportSection(title="Mechanistic", content="Test"),
        clinical_findings=ReportSection(title="Clinical", content="Test"),
        drug_candidates=["TestDrug"],
        limitations=["Test limitation"],
        conclusion="Test conclusion",
        references=[
            # Valid reference (matches sample_evidence)
            {
                "title": "Metformin mechanisms",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "authors": ["Smith J", "Jones A"],
                "date": "2023",
                "source": "pubmed"
            },
            # HALLUCINATED reference (URL doesn't exist in evidence)
            {
                "title": "Fake Paper That Doesn't Exist",
                "url": "https://fake-journal.com/made-up-paper",
                "authors": ["Hallucinated A"],
                "date": "2024",
                "source": "fake"
            },
            # Another HALLUCINATED reference
            {
                "title": "Invented Research",
                "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/",
                "authors": ["NotReal B"],
                "date": "2025",
                "source": "pubmed"
            }
        ],
        sources_searched=["pubmed"],
        total_papers_reviewed=1,
        search_iterations=1,
        confidence_score=0.5
    )

    # Validate - should remove hallucinated references
    validated_report = validate_references(report_with_hallucinations, sample_evidence)

    # Only the valid reference should remain
    assert len(validated_report.references) == 1
    assert validated_report.references[0]["title"] == "Metformin mechanisms"
    assert "Fake Paper" not in str(validated_report.references)


def test_citation_validator_handles_empty_references():
    """Citation validator should handle reports with no references."""
    from src.utils.citation_validator import validate_references

    report = ResearchReport(
        title="Empty Refs Report",
        executive_summary="This report has no references...",
        research_question="Testing empty refs",
        methodology=ReportSection(title="Methodology", content="Test"),
        hypotheses_tested=[],
        mechanistic_findings=ReportSection(title="Mechanistic", content="Test"),
        clinical_findings=ReportSection(title="Clinical", content="Test"),
        drug_candidates=[],
        limitations=[],
        conclusion="Test",
        references=[],  # Empty!
        sources_searched=[],
        total_papers_reviewed=0,
        search_iterations=0,
        confidence_score=0.0
    )

    validated = validate_references(report, [])
    assert validated.references == []
```

---

## 7. Definition of Done

Phase 8 is **COMPLETE** when:

1. `ResearchReport` model implemented with all sections
2. `ReportAgent` generates structured reports
3. Reports include proper citations and methodology
4. Magentic workflow uses ReportAgent for final synthesis
5. Report renders as clean markdown
6. All unit tests pass

---

## 8. Value Delivered

| Before (Phase 7) | After (Phase 8) |
|------------------|-----------------|
| Basic synthesis | Structured scientific report |
| Simple bullet points | Executive summary + methodology |
| List of citations | Formatted references |
| No methodology | Clear research process |
| No limitations | Honest limitations section |

**Sample output comparison:**

Before:
```
## Analysis
- Metformin might help
- Found 5 papers
[Link 1] [Link 2]
```

After:
```
# Drug Repurposing Analysis: Metformin for Alzheimer's Disease

## Executive Summary
Analysis of 15 papers suggests metformin may provide neuroprotection
through AMPK activation. Mechanistic evidence is strong (8/10),
while clinical evidence is moderate (6/10)...

## Methodology
Systematic search of PubMed and web sources using queries...

## Hypotheses Tested
- ✅ Metformin → AMPK → neuroprotection (7 supporting, 2 contradicting)

## References
1. Smith J, Jones A. *Metformin mechanisms*. Nature (2023). [Link](...)
```

---

## 9. Complete Magentic Architecture (Phases 5-8)

```
User Query
    ↓
Gradio UI
    ↓
Magentic Manager (LLM Coordinator)
    ├── SearchAgent ←→ PubMed + Web + VectorDB
    ├── HypothesisAgent ←→ Mechanistic Reasoning
    ├── JudgeAgent ←→ Evidence Assessment
    └── ReportAgent ←→ Final Synthesis
    ↓
Structured Research Report
```

**This matches Mario's diagram** with the practical agents that add real value for drug repurposing research.
