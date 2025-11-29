# SPEC 06: Simple Mode Synthesis Fix

## Priority: P0 (Blocker - Simple mode produces garbage output)

## Problem Statement

Simple mode (HuggingFace free tier) runs 10 iterations, collects 455 sources, but outputs only a citation dump with no actual synthesis. The user waits through the entire process just to see "Partial Analysis (Max Iterations Reached)" with no drug candidates or analysis.

**Observed Behavior** (real run):
```
Iterations 1-8:  confidence 70-90%, recommendation="continue"  ← Never synthesizes
Iteration 9-10:  confidence 0%                                 ← LLM context overflow
Final output:    Citation list only, no drug candidates, no analysis
```

---

## Research Context (November 2025 Best Practices)

This spec incorporates findings from current industry research on LLM-as-Judge, RAG systems, and multi-agent orchestration.

### LLM-as-Judge Biases ([Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge), [arXiv Survey](https://arxiv.org/abs/2411.15594))

| Bias | Description | Impact on Our System |
|------|-------------|---------------------|
| **Verbosity Bias** | LLM judges prefer longer, more detailed responses | Judge defaults to verbose "continue" explanations |
| **Position Bias** | Systematic preference based on order (primacy/recency) | Most recent evidence over-weighted |
| **Self-Preference Bias** | LLM favors outputs matching its own generation patterns | Defaults to "comfortable" pattern (continue) |

**Key Finding**: "Sophisticated judge models can align with human judgment up to 85%, which is actually higher than human-to-human agreement (81%)." However, this requires careful prompt design and debiasing.

### RAG Context Limits ([Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/), [TrueState](https://www.truestate.io/blog/lessons-from-rag))

> "Long context didn't kill retrieval. Bigger windows add cost and noise; **retrieval focuses attention where it matters.**"

**Key Finding**: RAG is **8-82× cheaper** than long context approaches. Best practice is:
- **Diverse selection** over recency-only selection
- **Re-ranking** before sending to judge
- **Lost-in-the-middle mitigation** - put critical context at prompt edges

### Multi-Agent Termination ([LangGraph Guide](https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025), [AWS Guidance](https://aws.amazon.com/solutions/guidance/multi-agent-orchestration-on-aws/))

> "The planning agent evaluates whether output **fully satisfies task objectives**. If so, the workflow is **terminated early**."

**Key Finding**: Code-enforced termination criteria outperform LLM-decided termination. The pattern is:
1. LLM provides **scores only** (mechanism, clinical, drug candidates)
2. Code evaluates scores against **explicit thresholds**
3. Code decides synthesize vs continue

---

## Root Cause Analysis

### Bug 1: No Evidence Limit in Judge Prompt (CRITICAL)

**File:** `src/prompts/judge.py:62`

```python
# BROKEN: Sends ALL evidence to the LLM
evidence_text = "\n\n".join([format_single_evidence(i, e) for i, e in enumerate(evidence)])
```

**Impact:**
- 455 sources × 1700 chars/source = **773,500 characters ≈ 193K tokens**
- HuggingFace Inference free tier limit: **~4K-8K tokens**
- Result: **Context overflow → LLM failure → fallback response → 0% confidence**

This explains why confidence dropped to 0% in iterations 9-10: the context became too large for the LLM.

### Bug 2: LLM Decides Both Scoring AND Recommendation (Anti-Pattern)

**Current Design:**
```python
# LLM does BOTH - subject to verbosity/self-preference bias
"Evaluate evidence... Respond with recommendation: 'continue' or 'synthesize'"
```

**Problem** (per 2025 research):
- LLM exhibits **self-preference bias** - defaults to its "comfortable" pattern
- "Be conservative" instruction triggers **verbosity bias** - prefers longer explanations for "continue"
- No **separation of concerns** - scoring and decision-making conflated

### Bug 3: No Diverse Evidence Selection

**Current Design:**
```python
# Just truncates to most recent - subject to position bias
capped_evidence = evidence[-30:]
```

**Problem** (per RAG research):
- **Position bias** - most recent ≠ most relevant
- **Lost-in-the-middle** - important early evidence ignored
- No **diversity** - may select 30 similar papers

### Bug 4: Prompt Encourages "Continue" Forever

**File:** `src/prompts/judge.py:22-32`

```python
## Sufficiency Criteria (TOO STRICT - requires ALL conditions)
- Combined scores >= 12 AND
- At least one specific drug candidate identified AND
- Clear mechanistic rationale exists

## Output Rules
- Be conservative: only recommend "synthesize" when truly confident  ← TRIGGERS VERBOSITY BIAS
```

### Bug 5: Search Derailment

**Evidence from logs:**
```
Next searches: androgen therapy and bone health, androgen therapy and muscle mass...
```

Original question: "female libido post-menopause" → Judge suggests tangentially related topics.

### Bug 6: Partial Synthesis is Garbage

**File:** `src/orchestrators/simple.py:432-470`

When max iterations reached, outputs only citations with no analysis, drug candidates, or key findings.

---

## Solution Design

### Architecture Change: Separate Scoring from Decision

**Before (biased):**
```
User Question → LLM Judge → { scores, recommendation } → Orchestrator follows recommendation
```

**After (debiased, per 2025 best practices):**
```
User Question → LLM Judge → { scores only } → Code evaluates → Code decides synthesize/continue
```

This follows the [Spring AI LLM-as-Judge pattern](https://spring.io/blog/2025/11/10/spring-ai-llm-as-judge-blog-post/): "Run agent in while loop with evaluator, until evaluator says output passes criteria" - but criteria are **code-enforced**, not LLM-decided.

---

### Fix 1: Diverse Evidence Selection (Not Just Capping)

**File:** `src/prompts/judge.py`

```python
MAX_EVIDENCE_FOR_JUDGE = 30  # Keep under token limits

async def select_evidence_for_judge(
    evidence: list[Evidence],
    query: str,
    max_items: int = MAX_EVIDENCE_FOR_JUDGE,
) -> list[Evidence]:
    """
    Select diverse, relevant evidence for judge evaluation.

    Implements RAG best practices (November 2025):
    - Diversity selection over recency-only
    - Lost-in-the-middle mitigation
    - Relevance re-ranking

    References:
    - https://www.pinecone.io/learn/retrieval-augmented-generation/
    - https://www.truestate.io/blog/lessons-from-rag
    """
    if len(evidence) <= max_items:
        return evidence

    try:
        from src.utils.text_utils import select_diverse_evidence
        # Use embedding-based diversity selection
        return await select_diverse_evidence(evidence, n=max_items, query=query)
    except ImportError:
        # Fallback: mix of recent + early (lost-in-the-middle mitigation)
        early = evidence[:max_items // 3]           # First third
        recent = evidence[-(max_items * 2 // 3):]   # Last two-thirds
        return early + recent


def format_user_prompt(
    question: str,
    evidence: list[Evidence],
    iteration: int = 0,
    max_iterations: int = 10,
    total_evidence_count: int | None = None,
) -> str:
    """
    Format user prompt with selected evidence and iteration context.

    NOTE: Evidence should be pre-selected using select_evidence_for_judge().
    This function assumes evidence is already capped.
    """
    total_count = total_evidence_count or len(evidence)
    max_content_len = 1500

    def format_single_evidence(i: int, e: Evidence) -> str:
        content = e.content
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        return (
            f"### Evidence {i + 1}\n"
            f"**Source**: {e.citation.source.upper()} - {e.citation.title}\n"
            f"**URL**: {e.citation.url}\n"
            f"**Content**:\n{content}"
        )

    evidence_text = "\n\n".join([format_single_evidence(i, e) for i, e in enumerate(evidence)])

    # Lost-in-the-middle mitigation: put critical context at START and END
    return f"""## Research Question (IMPORTANT - stay focused on this)
{question}

## Search Progress
- **Iteration**: {iteration}/{max_iterations}
- **Total evidence collected**: {total_count} sources
- **Evidence shown below**: {len(evidence)} diverse sources (selected for relevance)

## Available Evidence

{evidence_text}

## Your Task

Score this evidence for drug repurposing potential. Provide ONLY scores and extracted data.
DO NOT decide "synthesize" vs "continue" - that decision is made by the system.

## REMINDER: Original Question (stay focused)
{question}
"""
```

### Fix 2: Debiased Judge Prompt (Scoring Only)

**File:** `src/prompts/judge.py`

```python
SYSTEM_PROMPT = """You are an expert drug repurposing research judge.

Your task is to SCORE evidence from biomedical literature. You do NOT decide whether to
continue searching or synthesize - that decision is made by the orchestration system
based on your scores.

## Your Role: Scoring Only

You provide objective scores. The system decides next steps based on explicit thresholds.
This separation prevents bias in the decision-making process.

## Scoring Criteria

1. **Mechanism Score (0-10)**: How well does the evidence explain the biological mechanism?
   - 0-3: No clear mechanism, speculative
   - 4-6: Some mechanistic insight, but gaps exist
   - 7-10: Clear, well-supported mechanism of action

2. **Clinical Evidence Score (0-10)**: Strength of clinical/preclinical support?
   - 0-3: No clinical data, only theoretical
   - 4-6: Preclinical or early clinical data
   - 7-10: Strong clinical evidence (trials, meta-analyses)

3. **Drug Candidates**: List SPECIFIC drug names mentioned in the evidence
   - Only include drugs explicitly mentioned
   - Do NOT hallucinate or infer drug names
   - Include drug class if specific names aren't available (e.g., "SSRI antidepressants")

4. **Key Findings**: Extract 3-5 key findings from the evidence
   - Focus on findings relevant to the research question
   - Include mechanism insights and clinical outcomes

5. **Confidence (0.0-1.0)**: Your confidence in the scores
   - Based on evidence quality and relevance
   - Lower if evidence is tangential or low-quality

## Output Format

Return valid JSON with these fields:
- details.mechanism_score (int 0-10)
- details.mechanism_reasoning (string)
- details.clinical_evidence_score (int 0-10)
- details.clinical_reasoning (string)
- details.drug_candidates (list of strings)
- details.key_findings (list of strings)
- sufficient (boolean) - TRUE if scores suggest enough evidence
- confidence (float 0-1)
- recommendation ("continue" or "synthesize") - Your suggestion (system may override)
- next_search_queries (list) - If continuing, suggest FOCUSED queries
- reasoning (string)

## CRITICAL: Search Query Rules

When suggesting next_search_queries:
- STAY FOCUSED on the original research question
- Do NOT drift to tangential topics
- If question is about "female libido", do NOT suggest "bone health" or "muscle mass"
- Refine existing terms, don't explore random medical associations
- Example: "female libido post-menopause" → "testosterone therapy female sexual dysfunction"
"""
```

### Fix 3: Code-Enforced Termination Criteria

**File:** `src/orchestrators/simple.py`

```python
# Termination thresholds (code-enforced, not LLM-decided)
# Based on multi-agent orchestration best practices (November 2025)
# Reference: https://aws.amazon.com/solutions/guidance/multi-agent-orchestration-on-aws/

TERMINATION_CRITERIA = {
    "min_combined_score": 12,      # mechanism + clinical >= 12
    "min_score_with_volume": 10,   # >= 10 if 50+ sources
    "late_iteration_threshold": 8, # >= 8 in iterations 8+
    "max_evidence_threshold": 100, # Force synthesis with 100+ sources
    "emergency_iteration": 8,      # Last 2 iterations = emergency mode
    "min_confidence": 0.5,         # Minimum confidence for emergency synthesis
}


def should_synthesize(
    assessment: JudgeAssessment,
    iteration: int,
    max_iterations: int,
    evidence_count: int,
) -> tuple[bool, str]:
    """
    Code-enforced synthesis decision.

    Returns (should_synthesize, reason).

    This function implements the "explicit termination criteria" pattern
    from multi-agent orchestration best practices. The LLM provides scores,
    but CODE decides when to stop.

    Reference: https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025
    """
    combined_score = (
        assessment.details.mechanism_score +
        assessment.details.clinical_evidence_score
    )
    has_drug_candidates = len(assessment.details.drug_candidates) > 0
    confidence = assessment.confidence

    # Priority 1: LLM explicitly says sufficient with good scores
    if assessment.sufficient and assessment.recommendation == "synthesize":
        if combined_score >= 10:
            return True, "judge_approved"

    # Priority 2: High scores with drug candidates
    if combined_score >= TERMINATION_CRITERIA["min_combined_score"] and has_drug_candidates:
        return True, "high_scores_with_candidates"

    # Priority 3: Good scores with high evidence volume
    if combined_score >= TERMINATION_CRITERIA["min_score_with_volume"] and evidence_count >= 50:
        return True, "good_scores_high_volume"

    # Priority 4: Late iteration with acceptable scores (diminishing returns)
    is_late_iteration = iteration >= max_iterations - 2
    if is_late_iteration and combined_score >= TERMINATION_CRITERIA["late_iteration_threshold"]:
        return True, "late_iteration_acceptable"

    # Priority 5: Very high evidence count (enough to synthesize something)
    if evidence_count >= TERMINATION_CRITERIA["max_evidence_threshold"]:
        return True, "max_evidence_reached"

    # Priority 6: Emergency synthesis (avoid garbage output)
    if is_late_iteration and evidence_count >= 30 and confidence >= TERMINATION_CRITERIA["min_confidence"]:
        return True, "emergency_synthesis"

    return False, "continue_searching"
```

### Fix 4: Update Orchestrator Decision Phase

**File:** `src/orchestrators/simple.py`

```python
# In the run() method, replace the decision phase:

# === DECISION PHASE (Code-Enforced) ===
should_synth, reason = should_synthesize(
    assessment=assessment,
    iteration=iteration,
    max_iterations=self.config.max_iterations,
    evidence_count=len(all_evidence),
)

logger.info(
    "Synthesis decision",
    should_synthesize=should_synth,
    reason=reason,
    iteration=iteration,
    combined_score=assessment.details.mechanism_score + assessment.details.clinical_evidence_score,
    evidence_count=len(all_evidence),
    confidence=assessment.confidence,
)

if should_synth:
    # Log synthesis trigger reason for debugging
    if reason != "judge_approved":
        logger.info(f"Code-enforced synthesis triggered: {reason}")

    # Optional Analysis Phase
    async for event in self._run_analysis_phase(query, all_evidence, iteration):
        yield event

    yield AgentEvent(
        type="synthesizing",
        message=f"Evidence sufficient ({reason})! Preparing synthesis...",
        iteration=iteration,
    )

    # Generate final response
    final_response = self._generate_synthesis(query, all_evidence, assessment)

    yield AgentEvent(
        type="complete",
        message=final_response,
        data={
            "evidence_count": len(all_evidence),
            "iterations": iteration,
            "synthesis_reason": reason,
            "drug_candidates": assessment.details.drug_candidates,
            "key_findings": assessment.details.key_findings,
        },
        iteration=iteration,
    )
    return

else:
    # Need more evidence - prepare next queries
    current_queries = assessment.next_search_queries or [
        f"{query} mechanism of action",
        f"{query} clinical evidence",
    ]

    yield AgentEvent(
        type="looping",
        message=(
            f"Gathering more evidence (scores: {assessment.details.mechanism_score}+"
            f"{assessment.details.clinical_evidence_score}). "
            f"Next: {', '.join(current_queries[:2])}..."
        ),
        data={"next_queries": current_queries, "reason": reason},
        iteration=iteration,
    )
```

### Fix 5: Real Partial Synthesis

**File:** `src/orchestrators/simple.py`

```python
def _generate_partial_synthesis(
    self,
    query: str,
    evidence: list[Evidence],
) -> str:
    """
    Generate a REAL synthesis when max iterations reached.

    Even when forced to stop, we should provide:
    - Drug candidates (if any were found)
    - Key findings
    - Assessment scores
    - Actionable citations

    This is still better than a citation dump.
    """
    # Extract data from last assessment if available
    last_assessment = self.history[-1]["assessment"] if self.history else {}
    details = last_assessment.get("details", {})

    drug_candidates = details.get("drug_candidates", [])
    key_findings = details.get("key_findings", [])
    mechanism_score = details.get("mechanism_score", 0)
    clinical_score = details.get("clinical_evidence_score", 0)
    reasoning = last_assessment.get("reasoning", "Analysis incomplete due to iteration limit.")

    # Format drug candidates
    if drug_candidates:
        drug_list = "\n".join([f"- **{d}**" for d in drug_candidates[:5]])
    else:
        drug_list = "- *No specific drug candidates identified in evidence*\n- *Try a more specific query or add an API key for better analysis*"

    # Format key findings
    if key_findings:
        findings_list = "\n".join([f"- {f}" for f in key_findings[:5]])
    else:
        findings_list = "- *Key findings require further analysis*\n- *See citations below for relevant sources*"

    # Format citations (top 10)
    citations = "\n".join([
        f"{i + 1}. [{e.citation.title}]({e.citation.url}) "
        f"({e.citation.source.upper()}, {e.citation.date})"
        for i, e in enumerate(evidence[:10])
    ])

    combined_score = mechanism_score + clinical_score

    return f"""## Drug Repurposing Analysis

### Research Question
{query}

### Status
Analysis based on {len(evidence)} sources across {len(self.history)} iterations.
Maximum iterations reached - results may be incomplete.

### Drug Candidates Identified
{drug_list}

### Key Findings
{findings_list}

### Evidence Quality Scores
| Criterion | Score | Interpretation |
|-----------|-------|----------------|
| Mechanism | {mechanism_score}/10 | {"Strong" if mechanism_score >= 7 else "Moderate" if mechanism_score >= 4 else "Limited"} mechanistic evidence |
| Clinical | {clinical_score}/10 | {"Strong" if clinical_score >= 7 else "Moderate" if clinical_score >= 4 else "Limited"} clinical support |
| Combined | {combined_score}/20 | {"Sufficient" if combined_score >= 12 else "Partial"} for synthesis |

### Analysis Summary
{reasoning}

### Top Citations ({len(evidence)} sources total)
{citations}

---
*For more complete analysis:*
- *Add an OpenAI or Anthropic API key for enhanced LLM analysis*
- *Try a more specific query (e.g., include drug names)*
- *Use Advanced mode for multi-agent research*
"""
```

### Fix 6: Update Judge Handler Signature

**File:** `src/orchestrators/base.py`

```python
class JudgeHandlerProtocol(Protocol):
    """Protocol for judge handler."""

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
        iteration: int = 0,           # NEW
        max_iterations: int = 10,     # NEW
    ) -> JudgeAssessment:
        """Assess evidence quality and provide scores."""
        ...
```

**File:** `src/agent_factory/judges.py`

Update all handlers (`JudgeHandler`, `HFInferenceJudgeHandler`, `MockJudgeHandler`) to:

```python
async def assess(
    self,
    question: str,
    evidence: list[Evidence],
    iteration: int = 0,
    max_iterations: int = 10,
) -> JudgeAssessment:
    """Assess evidence with iteration context."""
    # Select diverse evidence (not just truncate)
    selected_evidence = await select_evidence_for_judge(evidence, question)

    # Format prompt with iteration context
    user_prompt = format_user_prompt(
        question=question,
        evidence=selected_evidence,
        iteration=iteration,
        max_iterations=max_iterations,
        total_evidence_count=len(evidence),
    )

    # ... rest of implementation
```

---

## Implementation Order

| Order | Fix | Priority | Impact |
|-------|-----|----------|--------|
| 1 | Diverse evidence selection | CRITICAL | Prevents token overflow + position bias |
| 2 | Code-enforced termination | CRITICAL | Guarantees synthesis before max iterations |
| 3 | Debiased judge prompt | HIGH | Removes verbosity/self-preference bias |
| 4 | Real partial synthesis | HIGH | Ensures useful output even on forced stop |
| 5 | Update handler signatures | MEDIUM | Enables iteration context |
| 6 | Update orchestrator | MEDIUM | Integrates all fixes |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/prompts/judge.py` | New `select_evidence_for_judge()`, updated `format_user_prompt()`, debiased `SYSTEM_PROMPT` |
| `src/orchestrators/simple.py` | New `should_synthesize()`, updated decision phase, real `_generate_partial_synthesis()` |
| `src/orchestrators/base.py` | Update `JudgeHandlerProtocol` signature |
| `src/agent_factory/judges.py` | Update all handlers with iteration params, use diverse selection |

---

## Test Plan

### Unit Tests

```python
# tests/unit/prompts/test_judge_prompt.py

@pytest.mark.asyncio
async def test_evidence_selection_diverse():
    """Verify evidence selection includes early and recent items."""
    evidence = [make_evidence(f"Paper {i}") for i in range(100)]
    selected = await select_evidence_for_judge(evidence, "test query", max_items=30)

    # Should include some early evidence (lost-in-the-middle mitigation)
    titles = [e.citation.title for e in selected]
    assert any("Paper 0" in t or "Paper 1" in t for t in titles)
    assert any("Paper 99" in t or "Paper 98" in t for t in titles)


def test_prompt_includes_question_at_edges():
    """Verify lost-in-the-middle mitigation."""
    evidence = [make_evidence("Test")]
    prompt = format_user_prompt("important question", evidence, iteration=5, max_iterations=10)

    # Question should appear at START and END of prompt
    lines = prompt.split("\n")
    assert "important question" in lines[1]  # Near start
    assert "important question" in lines[-2]  # Near end


# tests/unit/orchestrators/test_termination.py

def test_should_synthesize_high_scores():
    """High scores with drug candidates triggers synthesis."""
    assessment = make_assessment(mechanism=7, clinical=6, drug_candidates=["Metformin"])
    should_synth, reason = should_synthesize(assessment, iteration=3, max_iterations=10, evidence_count=50)

    assert should_synth is True
    assert reason == "high_scores_with_candidates"


def test_should_synthesize_late_iteration():
    """Late iteration with acceptable scores triggers synthesis."""
    assessment = make_assessment(mechanism=5, clinical=4, drug_candidates=[])
    should_synth, reason = should_synthesize(assessment, iteration=9, max_iterations=10, evidence_count=80)

    assert should_synth is True
    assert reason in ["late_iteration_acceptable", "emergency_synthesis"]


def test_should_not_synthesize_early_low_scores():
    """Early iteration with low scores continues searching."""
    assessment = make_assessment(mechanism=3, clinical=2, drug_candidates=[])
    should_synth, reason = should_synthesize(assessment, iteration=2, max_iterations=10, evidence_count=20)

    assert should_synth is False
    assert reason == "continue_searching"


def test_partial_synthesis_has_drug_candidates():
    """Partial synthesis includes extracted data."""
    orchestrator = Orchestrator(...)
    orchestrator.history = [{
        "assessment": {
            "details": {
                "drug_candidates": ["Testosterone", "DHEA"],
                "key_findings": ["Finding 1", "Finding 2"],
                "mechanism_score": 6,
                "clinical_evidence_score": 5,
            },
            "reasoning": "Good evidence found.",
        }
    }]

    result = orchestrator._generate_partial_synthesis("test", [make_evidence("Test")])

    assert "Testosterone" in result
    assert "DHEA" in result
    assert "Drug Candidates" in result
    assert "6/10" in result  # mechanism score
```

### Integration Tests

```python
# tests/integration/test_simple_mode_synthesis.py

@pytest.mark.asyncio
async def test_simple_mode_synthesizes_before_max_iterations():
    """Verify simple mode produces useful output with mocked judge."""
    # Mock judge to return good scores
    mock_judge = MockJudgeHandler()
    orchestrator = Orchestrator(
        search_handler=mock_search_handler,
        judge_handler=mock_judge,
    )

    events = []
    async for event in orchestrator.run("metformin diabetes mechanism"):
        events.append(event)

    # Must have synthesis with drug candidates
    complete_event = next(e for e in events if e.type == "complete")
    assert "Drug Candidates" in complete_event.message
    assert complete_event.data.get("synthesis_reason") is not None


@pytest.mark.asyncio
async def test_large_evidence_does_not_crash():
    """Verify 500 sources don't cause token overflow."""
    evidence = [make_evidence(f"Paper {i}") for i in range(500)]
    selected = await select_evidence_for_judge(evidence, "test query")

    # Should be capped
    assert len(selected) <= MAX_EVIDENCE_FOR_JUDGE

    # Total chars should be under ~50K (safe for most LLMs)
    prompt = format_user_prompt("test", selected, iteration=5, max_iterations=10, total_evidence_count=500)
    assert len(prompt) < 100_000  # Well under token limits
```

---

## Acceptance Criteria

- [ ] Evidence sent to judge is diverse-selected (not just truncated)
- [ ] Prompt includes question at START and END (lost-in-the-middle mitigation)
- [ ] Code-enforced `should_synthesize()` makes termination decision
- [ ] Synthesis triggered by iteration 8 with 50+ sources and scores >= 8
- [ ] Partial synthesis includes drug candidates and scores (not just citations)
- [ ] Search queries stay on-topic (judge prompt enforces focus)
- [ ] 500+ sources don't cause LLM crashes
- [ ] All existing tests pass

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Diverse selection misses critical evidence | Include relevance scoring in selection |
| Code-enforced thresholds too aggressive | Log all synthesis decisions for tuning |
| Prompt changes affect OpenAI/Anthropic differently | Test with all providers |
| Emergency synthesis produces low-quality output | Still better than citation dump |

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Synthesis rate | 0% | 90%+ |
| Average iterations to synthesis | 10 (max) | 5-7 |
| Drug candidates in output | Never | Always (if found) |
| LLM token overflow errors | Common | None |
| User-reported "useless output" | Frequent | Rare |

---

## References

- [LLM-as-a-Judge Guide - Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Survey on LLM-as-a-Judge - arXiv](https://arxiv.org/abs/2411.15594)
- [RAG Best Practices - Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Lessons from RAG 2025 - TrueState](https://www.truestate.io/blog/lessons-from-rag)
- [LangGraph Multi-Agent Orchestration 2025](https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
- [Multi-Agent Orchestration on AWS](https://aws.amazon.com/solutions/guidance/multi-agent-orchestration-on-aws/)
- [Spring AI LLM-as-Judge Pattern](https://spring.io/blog/2025/11/10/spring-ai-llm-as-judge-blog-post/)
