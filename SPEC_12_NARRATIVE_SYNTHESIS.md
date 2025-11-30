# SPEC_12: Narrative Report Synthesis

**Status**: Ready for Implementation
**Priority**: P1 - Core deliverable
**Related Issues**: #85, #86
**Related Spec**: SPEC_11 (Sexual Health Focus)
**Author**: Deep Audit against Microsoft Agent Framework

---

## Problem Statement

DeepBoner's report generation outputs **structured metadata** instead of **synthesized prose**. The current implementation uses string templating with NO LLM call for narrative synthesis.

### Current Output (Simple Mode - What Users See)

```markdown
## Sexual Health Analysis

### Question
Testosterone therapy for hypoactive sexual desire disorder?

### Drug Candidates
- **Testosterone**
- **LibiGel**

### Key Findings
- Testosterone therapy improves sexual desire

### Assessment
- **Mechanism Score**: 8/10
- **Clinical Evidence Score**: 9/10
- **Confidence**: 90%

### Citations (33 sources)
1. [Title](url)...
```

### Expected Output (Professional Research Report)

```markdown
## Sexual Health Research Report: Testosterone Therapy for HSDD

### Executive Summary

Testosterone therapy represents a well-established, evidence-based treatment for
hypoactive sexual desire disorder (HSDD) in postmenopausal women. Our analysis of
33 peer-reviewed sources reveals consistent findings across multiple randomized
controlled trials, with transdermal testosterone demonstrating the strongest
efficacy-safety profile.

### Background

Hypoactive sexual desire disorder affects an estimated 12% of postmenopausal women
and is characterized by persistent lack of sexual interest causing personal distress.
The ISSWSH published clinical guidelines in 2021 establishing testosterone as a
recommended intervention...

### Evidence Synthesis

**Mechanism of Action**

Testosterone exerts its effects on sexual desire through multiple pathways. At the
hypothalamic level, testosterone modulates dopaminergic signaling. Evidence from
Smith et al. (2021) demonstrates androgen receptor activation correlates with
subjective measures of desire (r=0.67, p<0.001)...

### Recommendations

1. **Transdermal testosterone** (300 μg/day) is recommended for postmenopausal
   women with HSDD not primarily related to modifiable factors
2. **Duration**: Continue for 6 months to assess efficacy; discontinue if no benefit

### Limitations

Long-term safety data beyond 24 months remains limited...

### References
1. Smith AB et al. (2021). Testosterone mechanisms... https://pubmed.ncbi.nlm.nih.gov/123/
```

---

## Root Cause Analysis

### Location 1: Simple Orchestrator (THE PRIMARY BUG)

**File**: `src/orchestrators/simple.py`
**Lines**: 448-505
**Method**: `_generate_synthesis()`

```python
def _generate_synthesis(
    self,
    query: str,
    evidence: list[Evidence],
    assessment: JudgeAssessment,
) -> str:
    # ❌ NO LLM CALL - Just string templating!
    drug_list = "\n".join([f"- **{d}**" for d in assessment.details.drug_candidates])
    findings_list = "\n".join([f"- {f}" for f in assessment.details.key_findings])

    return f"""{self.domain_config.report_title}
### Question
{query}
### Drug Candidates
{drug_list}
...
"""
```

**The Problem**: No LLM is ever called. It's just formatted data from JudgeAssessment.

### Location 2: Partial Synthesis (Max Iterations Fallback)

**File**: `src/orchestrators/simple.py`
**Lines**: 507-602
**Method**: `_generate_partial_synthesis()`

Same issue - string templating, no LLM call.

### Location 3: Report Agent (Advanced Mode)

**File**: `src/agents/report_agent.py`
**Lines**: 93-94

```python
result = await self._get_agent().run(prompt)
report = result.output  # ResearchReport (structured data)
```

This DOES make an LLM call, but it outputs `ResearchReport` (structured Pydantic model), not narrative prose. The `to_markdown()` method just formats the structured fields.

### Location 4: Report System Prompt

**File**: `src/prompts/report.py`
**Lines**: 13-76

The system prompt tells the LLM to output structured JSON with fields like `hypotheses_tested: [...]` and `references: [...]`. It does NOT request narrative prose.

---

## Microsoft Agent Framework Pattern (Reference)

**File**: `reference_repos/agent-framework/python/samples/getting_started/workflows/orchestration/concurrent_custom_aggregator.py`
**Lines**: 56-79

```python
# Define a custom aggregator callback that uses the chat client to SYNTHESIZE
async def summarize_results(results: list[Any]) -> str:
    expert_sections: list[str] = []
    for r in results:
        messages = getattr(r.agent_run_response, "messages", [])
        final_text = messages[-1].text if messages else "(no content)"
        expert_sections.append(f"{r.executor_id}:\n{final_text}")

    # ✅ LLM CALL for synthesis
    system_msg = ChatMessage(
        Role.SYSTEM,
        text=(
            "You are a helpful assistant that consolidates multiple domain expert outputs "
            "into one cohesive, concise summary with clear takeaways."
        ),
    )
    user_msg = ChatMessage(Role.USER, text="\n\n".join(expert_sections))

    response = await chat_client.get_response([system_msg, user_msg])
    return response.messages[-1].text
```

**The pattern**: The aggregator makes an **LLM call** to synthesize, not string concatenation.

---

## Solution Design

### Architecture Change

```text
Current (Simple Mode):
  Evidence → Judge → {structured data} → String Template → Bullet Points

Proposed (Simple Mode):
  Evidence → Judge → {structured data} → LLM Synthesis → Narrative Prose
                                              ↓
                                     Uses SynthesisPrompt
```

### Components to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/prompts/synthesis.py` | **NEW** | Narrative synthesis prompts |
| `src/orchestrators/simple.py` | **MODIFY** | Make `_generate_synthesis()` async, add LLM call |
| `src/config/domain.py` | **MODIFY** | Add `synthesis_system_prompt` field |
| `tests/unit/prompts/test_synthesis.py` | **NEW** | Test synthesis prompts |
| `tests/unit/orchestrators/test_simple_synthesis.py` | **NEW** | Test LLM synthesis |

---

## Implementation Plan

### Phase 1: Create Synthesis Prompts

**File**: `src/prompts/synthesis.py` (NEW)

```python
"""Prompts for narrative report synthesis."""

from src.config.domain import ResearchDomain, get_domain_config

def get_synthesis_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for narrative synthesis."""
    config = get_domain_config(domain)
    return f"""You are a scientific writer specializing in {config.name.lower()}.
Your task is to synthesize research evidence into a clear, NARRATIVE report.

## CRITICAL: Writing Style
- Write in PROSE PARAGRAPHS, not bullet points
- Use academic but accessible language
- Be specific about evidence strength (e.g., "in an RCT of N=200")
- Reference specific studies by author name
- Provide quantitative results where available (p-values, effect sizes)

## Report Structure

### Executive Summary (REQUIRED - 2-3 sentences)
Start with the bottom line. Example:
"Testosterone therapy demonstrates consistent efficacy for HSDD in postmenopausal
women, with transdermal formulations showing the best safety profile."

### Background (REQUIRED - 1 paragraph)
Explain the condition, its prevalence, and clinical significance.

### Evidence Synthesis (REQUIRED - 2-4 paragraphs)
Weave the evidence into a coherent NARRATIVE:
- Mechanism of Action: How does the intervention work?
- Clinical Evidence: What do trials show? Include effect sizes.
- Comparative Evidence: How does it compare to alternatives?

### Recommendations (REQUIRED - 3-5 items)
Provide actionable clinical recommendations.

### Limitations (REQUIRED - 1 paragraph)
Acknowledge gaps, biases, and areas needing more research.

### References (REQUIRED)
List key references with author, year, title, URL.

## CRITICAL RULES
1. ONLY cite papers from the provided evidence - NEVER hallucinate references
2. Write in complete sentences and paragraphs (PROSE, not lists)
3. Include specific statistics when available
4. Acknowledge uncertainty honestly
"""


FEW_SHOT_EXAMPLE = '''
## Example: Strong Evidence Synthesis

INPUT:
- Query: "Alprostadil for erectile dysfunction"
- Evidence: 15 papers including meta-analysis of 8 RCTs (N=3,247)
- Mechanism Score: 9/10
- Clinical Score: 9/10

OUTPUT:

### Executive Summary

Alprostadil (prostaglandin E1) represents a well-established second-line treatment
for erectile dysfunction, with meta-analytic evidence demonstrating 87% efficacy
in achieving erections sufficient for intercourse. It offers a PDE5-independent
mechanism particularly valuable for patients who do not respond to oral therapies.

### Background

Erectile dysfunction affects approximately 30 million men in the United States,
with prevalence increasing with age. While PDE5 inhibitors remain first-line
therapy, approximately 30% of patients are non-responders. Alprostadil provides
an alternative mechanism through direct smooth muscle relaxation.

### Evidence Synthesis

**Mechanism of Action**

Alprostadil works through a distinct pathway from PDE5 inhibitors. It binds to
EP receptors on cavernosal smooth muscle, activating adenylate cyclase and
increasing intracellular cAMP. As noted by Smith et al. (2019), this mechanism
explains its efficacy in patients with endothelial dysfunction.

**Clinical Evidence**

A meta-analysis by Johnson et al. (2020) pooled data from 8 randomized controlled
trials (N=3,247). The primary endpoint of erection sufficient for intercourse was
achieved in 87% of alprostadil patients versus 12% placebo (RR 7.25, 95% CI:
5.8-9.1, p<0.001). The NNT was 1.3, indicating robust effect size.

### Recommendations

1. Consider alprostadil as second-line therapy when PDE5 inhibitors fail
2. Start with 10 μg intracavernosal injection, titrate to 40 μg
3. Provide in-office training for self-injection technique

### Limitations

Long-term data beyond 2 years is limited. Head-to-head comparisons with newer
therapies are lacking. Most trials excluded severe cardiovascular disease.

### References

1. Smith AB et al. (2019). Alprostadil mechanism. J Urol. https://pubmed.ncbi.nlm.nih.gov/123/
2. Johnson CD et al. (2020). Meta-analysis of alprostadil. J Sex Med. https://pubmed.ncbi.nlm.nih.gov/456/
'''


def format_synthesis_prompt(
    query: str,
    evidence_summary: str,
    drug_candidates: list[str],
    key_findings: list[str],
    mechanism_score: int,
    clinical_score: int,
    confidence: float,
) -> str:
    """Format the user prompt for synthesis."""
    return f"""Synthesize a narrative research report for the following query.

## Research Question
{query}

## Evidence Summary
{evidence_summary}

## Identified Drug Candidates
{', '.join(drug_candidates) or 'None identified'}

## Key Findings from Evidence
{chr(10).join(f'- {f}' for f in key_findings) or 'No specific findings'}

## Assessment Scores
- Mechanism Score: {mechanism_score}/10
- Clinical Evidence Score: {clinical_score}/10
- Confidence: {confidence:.0%}

## Instructions
Generate a NARRATIVE research report following the structure above.
Write in prose paragraphs, NOT bullet points (except for Recommendations).
ONLY cite papers mentioned in the Evidence Summary above.

{FEW_SHOT_EXAMPLE}
"""
```

### Phase 2: Update Simple Orchestrator

**File**: `src/orchestrators/simple.py`
**Change**: Make `_generate_synthesis()` async and add LLM call

```python
# Add imports at top
from src.prompts.synthesis import get_synthesis_system_prompt, format_synthesis_prompt
from src.agent_factory.judges import get_model
from pydantic_ai import Agent

# Change method signature and implementation (lines 448-505)
async def _generate_synthesis(
    self,
    query: str,
    evidence: list[Evidence],
    assessment: JudgeAssessment,
) -> str:
    """
    Generate the final synthesis response using LLM.

    Args:
        query: The original question
        evidence: All collected evidence
        assessment: The final assessment

    Returns:
        Narrative synthesis as markdown
    """
    # Build evidence summary for LLM context
    evidence_lines = []
    for e in evidence[:20]:  # Limit context
        authors = ", ".join(e.citation.authors[:2]) if e.citation.authors else "Unknown"
        evidence_lines.append(
            f"- {e.citation.title} ({authors}, {e.citation.date}): {e.content[:200]}..."
        )
    evidence_summary = "\n".join(evidence_lines)

    # Format synthesis prompt
    user_prompt = format_synthesis_prompt(
        query=query,
        evidence_summary=evidence_summary,
        drug_candidates=assessment.details.drug_candidates,
        key_findings=assessment.details.key_findings,
        mechanism_score=assessment.details.mechanism_score,
        clinical_score=assessment.details.clinical_evidence_score,
        confidence=assessment.confidence,
    )

    # Create synthesis agent
    system_prompt = get_synthesis_system_prompt(self.domain)

    try:
        agent: Agent[None, str] = Agent(
            model=get_model(),
            output_type=str,
            system_prompt=system_prompt,
        )
        result = await agent.run(user_prompt)
        narrative = result.output
    except Exception as e:
        # Fallback to template if LLM fails
        logger.warning("LLM synthesis failed, using template", error=str(e))
        return self._generate_template_synthesis(query, evidence, assessment)

    # Add citations footer
    citations = "\n".join(
        f"{i + 1}. [{e.citation.title}]({e.citation.url}) "
        f"({e.citation.source.upper()}, {e.citation.date})"
        for i, e in enumerate(evidence[:10])
    )

    return f"""{narrative}

---
### Full Citation List ({len(evidence)} sources)
{citations}

*Analysis based on {len(evidence)} sources across {len(self.history)} iterations.*
"""

def _generate_template_synthesis(
    self,
    query: str,
    evidence: list[Evidence],
    assessment: JudgeAssessment,
) -> str:
    """Fallback template synthesis (no LLM)."""
    # Keep the existing string template logic here as fallback
    ...
```

### Phase 3: Update Call Site

**File**: `src/orchestrators/simple.py`
**Line**: 393

```python
# Change from:
final_response = self._generate_synthesis(query, all_evidence, assessment)

# To:
final_response = await self._generate_synthesis(query, all_evidence, assessment)
```

### Phase 4: Update Domain Config

**File**: `src/config/domain.py`

Add optional `synthesis_system_prompt` field to `DomainConfig`:

```python
class DomainConfig(BaseModel):
    # ... existing fields ...

    # Synthesis (optional, can inherit from base)
    synthesis_system_prompt: str | None = None
```

### Phase 5: Add Tests

**File**: `tests/unit/prompts/test_synthesis.py` (NEW)

```python
"""Tests for synthesis prompts."""

import pytest

from src.prompts.synthesis import (
    get_synthesis_system_prompt,
    format_synthesis_prompt,
    FEW_SHOT_EXAMPLE,
)


def test_synthesis_system_prompt_is_narrative_focused() -> None:
    """System prompt should emphasize prose, not bullets."""
    prompt = get_synthesis_system_prompt()
    assert "PROSE PARAGRAPHS" in prompt
    assert "not bullet points" in prompt.lower()
    assert "Executive Summary" in prompt


def test_synthesis_system_prompt_warns_about_hallucination() -> None:
    """System prompt should warn about citation hallucination."""
    prompt = get_synthesis_system_prompt()
    assert "NEVER hallucinate" in prompt


def test_format_synthesis_prompt_includes_evidence() -> None:
    """User prompt should include evidence summary."""
    prompt = format_synthesis_prompt(
        query="testosterone libido",
        evidence_summary="Study shows efficacy...",
        drug_candidates=["Testosterone"],
        key_findings=["Improved libido"],
        mechanism_score=8,
        clinical_score=7,
        confidence=0.85,
    )
    assert "testosterone libido" in prompt
    assert "Study shows efficacy" in prompt
    assert "Testosterone" in prompt
    assert "8/10" in prompt


def test_few_shot_example_is_narrative() -> None:
    """Few-shot example should demonstrate narrative style."""
    # Count paragraphs vs bullets
    paragraphs = len([p for p in FEW_SHOT_EXAMPLE.split('\n\n') if len(p) > 100])
    bullets = FEW_SHOT_EXAMPLE.count('\n- ')

    # Prose should dominate (at least 2x more paragraphs than bullets)
    assert paragraphs >= bullets, "Few-shot example should be mostly narrative"
```

**File**: `tests/unit/orchestrators/test_simple_synthesis.py` (NEW)

```python
"""Tests for simple orchestrator synthesis."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestrators.simple import Orchestrator
from src.utils.models import Evidence, Citation, JudgeAssessment, JudgeDetails


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            content="Testosterone therapy shows efficacy in HSDD treatment.",
            citation=Citation(
                source="pubmed",
                title="Testosterone and Female Libido",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2023",
                authors=["Smith J"],
            ),
        )
    ]


@pytest.fixture
def sample_assessment() -> JudgeAssessment:
    return JudgeAssessment(
        sufficient=True,
        confidence=0.85,
        reasoning="Evidence is sufficient",
        recommendation="synthesize",
        next_search_queries=[],
        details=JudgeDetails(
            mechanism_score=8,
            clinical_evidence_score=7,
            drug_candidates=["Testosterone"],
            key_findings=["Improved libido in postmenopausal women"],
        ),
    )


@pytest.mark.asyncio
async def test_generate_synthesis_calls_llm(
    sample_evidence: list[Evidence],
    sample_assessment: JudgeAssessment,
) -> None:
    """Synthesis should make an LLM call, not just template."""
    mock_search = MagicMock()
    mock_judge = MagicMock()

    orchestrator = Orchestrator(
        search_handler=mock_search,
        judge_handler=mock_judge,
    )

    with patch("src.orchestrators.simple.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "This is a narrative synthesis with prose paragraphs."
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_agent

        result = await orchestrator._generate_synthesis(
            query="testosterone HSDD",
            evidence=sample_evidence,
            assessment=sample_assessment,
        )

        # Verify LLM was called
        mock_agent_class.assert_called_once()
        mock_agent.run.assert_called_once()

        # Verify output includes narrative
        assert "narrative synthesis" in result.lower() or "prose" in result.lower()


@pytest.mark.asyncio
async def test_generate_synthesis_falls_back_on_error(
    sample_evidence: list[Evidence],
    sample_assessment: JudgeAssessment,
) -> None:
    """Synthesis should fall back to template if LLM fails."""
    mock_search = MagicMock()
    mock_judge = MagicMock()

    orchestrator = Orchestrator(
        search_handler=mock_search,
        judge_handler=mock_judge,
    )

    with patch("src.orchestrators.simple.Agent") as mock_agent_class:
        mock_agent_class.side_effect = Exception("LLM unavailable")

        result = await orchestrator._generate_synthesis(
            query="testosterone HSDD",
            evidence=sample_evidence,
            assessment=sample_assessment,
        )

        # Should still return something (template fallback)
        assert "Sexual Health Analysis" in result or "testosterone" in result.lower()
```

---

## File Changes Summary

| File | Lines | Change Type | Description |
|------|-------|-------------|-------------|
| `src/prompts/synthesis.py` | ~150 | NEW | Narrative synthesis prompts |
| `src/orchestrators/simple.py` | 393, 448-505 | MODIFY | Async synthesis with LLM |
| `src/config/domain.py` | 57 | MODIFY | Add `synthesis_system_prompt` |
| `tests/unit/prompts/test_synthesis.py` | ~60 | NEW | Prompt tests |
| `tests/unit/orchestrators/test_simple_synthesis.py` | ~80 | NEW | Synthesis tests |

---

## Acceptance Criteria

- [ ] Report contains **paragraph-form prose**, not just bullet points
- [ ] Report has **executive summary** (2-3 sentences)
- [ ] Report has **background section** explaining the condition
- [ ] Report has **synthesized narrative** weaving evidence together
- [ ] Report has **actionable recommendations**
- [ ] Report has **limitations** section
- [ ] Citations are **properly formatted** (author, year, title, URL)
- [ ] No hallucinated references (CRITICAL)
- [ ] Falls back gracefully if LLM unavailable
- [ ] All existing tests still pass
- [ ] New tests achieve 90%+ coverage of synthesis code

---

## Test Criteria

```python
def test_report_is_narrative_not_bullets():
    """Report should be mostly prose, not bullet points."""
    report = await orchestrator._generate_synthesis(...)

    # Count paragraphs vs bullet points
    paragraphs = len([p for p in report.split('\n\n') if len(p) > 100])
    bullets = report.count('\n- ')

    # Prose should dominate
    assert paragraphs > bullets, "Report should be narrative, not bullet list"

def test_references_not_hallucinated():
    """All references must come from provided evidence."""
    evidence_urls = {e.citation.url for e in evidence}
    report = await orchestrator._generate_synthesis(...)

    # Extract URLs from report
    import re
    report_urls = set(re.findall(r'https?://[^\s\)]+', report))

    for url in report_urls:
        # Allow pubmed URLs even if slightly different format
        if "pubmed" in url or "clinicaltrials" in url:
            assert any(evidence_url in url or url in evidence_url
                      for evidence_url in evidence_urls), f"Hallucinated: {url}"
```

---

## Related Microsoft Agent Framework Patterns

| Pattern | File | Application |
|---------|------|-------------|
| Custom Aggregator | `concurrent_custom_aggregator.py:56-79` | LLM-based synthesis |
| Fan-Out/Fan-In | `fan_out_fan_in_edges.py` | Multi-expert synthesis |
| Sequential Chain | `sequential_agents.py` | Writer→Reviewer pattern |

---

## Implementation Notes for Async Agent

1. **Start with `src/prompts/synthesis.py`** - This is independent and can be created first
2. **Then modify `src/orchestrators/simple.py`** - Change `_generate_synthesis` to async
3. **Update the call site** (line 393) - Add `await`
4. **Add tests** - Both unit and integration
5. **Run `make check`** - Ensure all 237+ tests still pass

The key insight from the MS Agent Framework is:
> The aggregator makes an **LLM call** to synthesize, not string concatenation.

Our `_generate_synthesis()` currently does NO LLM call. Fix that, and the reports will transform from bullet points to narrative prose.

---

## References

- GitHub Issue #85: Report lacks narrative synthesis
- GitHub Issue #86: Microsoft Agent Framework patterns
- `reference_repos/agent-framework/python/samples/getting_started/workflows/orchestration/concurrent_custom_aggregator.py`
- LangChain Deep Agents: Few-shot examples importance
