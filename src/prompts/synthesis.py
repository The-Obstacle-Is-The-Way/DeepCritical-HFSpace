"""Prompts for narrative report synthesis.

This module provides prompts that transform structured evidence data
into professional, narrative research reports. The key insight is that
report generation requires an LLM call for synthesis, not string templating.

Reference: Microsoft Agent Framework concurrent_custom_aggregator.py pattern.
"""

from src.config.domain import ResearchDomain, get_domain_config


def get_synthesis_system_prompt(domain: ResearchDomain | str | None = None) -> str:
    """Get the system prompt for narrative synthesis.

    Args:
        domain: Research domain for customization (defaults to settings)

    Returns:
        System prompt instructing LLM to write narrative prose
    """
    config = get_domain_config(domain)
    return f"""You are a scientific writer specializing in {config.name.lower()}.
Your task is to synthesize research evidence into a clear, NARRATIVE report.

## CRITICAL: Writing Style
- Write in PROSE PARAGRAPHS, not bullet points
- Use academic but accessible language
- Be specific about evidence strength (e.g., "in an RCT of N=200")
- Reference specific studies by author name when available
- Provide quantitative results where available (p-values, effect sizes, NNT)

## Report Structure

### Executive Summary (REQUIRED - 2-3 sentences)
Start with the bottom line. What does the evidence show? Example:
"Testosterone therapy demonstrates consistent efficacy for HSDD in postmenopausal
women, with transdermal formulations showing the best safety profile."

### Background (REQUIRED - 1 paragraph)
Explain the condition, its prevalence, and clinical significance.
Why does this question matter?

### Evidence Synthesis (REQUIRED - 2-4 paragraphs)
Weave the evidence into a coherent NARRATIVE:
- **Mechanism of Action**: How does the intervention work biologically?
- **Clinical Evidence**: What do trials show? Include effect sizes when available.
- **Comparative Evidence**: How does it compare to alternatives?

Write this as flowing prose that tells a story, NOT as a bullet list.

### Recommendations (REQUIRED - 3-5 numbered items)
Provide specific, actionable clinical recommendations based on the evidence.
These CAN be numbered items since they are action items.

### Limitations (REQUIRED - 1 paragraph)
Acknowledge gaps in the evidence, potential biases, and areas needing more research.
Be honest about uncertainty.

### References (REQUIRED)
List key references with author, year, title, and URL.
Format: Author AB et al. (Year). Title. URL

## CRITICAL RULES
1. ONLY cite papers from the provided evidence - NEVER hallucinate or invent references
2. Write in complete sentences and paragraphs (PROSE, not lists except Recommendations)
3. Include specific statistics when available (p-values, confidence intervals, effect sizes)
4. Acknowledge uncertainty honestly - do not overstate conclusions
5. If evidence is limited, say so clearly
6. Copy URLs exactly as provided - do not create similar-looking URLs
"""


FEW_SHOT_EXAMPLE = """
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
with prevalence increasing with age from 12% at age 40 to 40% at age 70. While
PDE5 inhibitors remain first-line therapy, approximately 30% of patients are
non-responders due to diabetes, radical prostatectomy, or other factors.
Alprostadil provides an alternative mechanism through direct smooth muscle
relaxation, making it a crucial second-line option.

### Evidence Synthesis

**Mechanism of Action**

Alprostadil works through a distinct pathway from PDE5 inhibitors. It binds to
EP2 and EP4 receptors on cavernosal smooth muscle, activating adenylate cyclase
and increasing intracellular cAMP. This leads to smooth muscle relaxation and
increased blood flow independent of nitric oxide signaling. As noted by Smith
et al. (2019), this mechanism explains its efficacy in patients with endothelial
dysfunction where nitric oxide production is impaired.

**Clinical Evidence**

A meta-analysis by Johnson et al. (2020) pooled data from 8 randomized controlled
trials (N=3,247). The primary endpoint of erection sufficient for intercourse was
achieved in 87% of alprostadil patients versus 12% placebo (RR 7.25, 95% CI:
5.8-9.1, p<0.001). The number needed to treat was 1.3, indicating robust effect
size. Onset of action was 5-15 minutes, with duration of 30-60 minutes.

**Comparative Evidence**

Direct comparisons with PDE5 inhibitors are limited. However, in the subgroup
of PDE5 non-responders studied by Martinez et al. (2018), alprostadil achieved
successful intercourse in 72% of patients who had failed sildenafil.

### Recommendations

1. Consider alprostadil as second-line therapy when PDE5 inhibitors fail or are
   contraindicated
2. Start with 10 micrograms intracavernosal injection, titrate to 40 micrograms based
   on response
3. Provide in-office training for self-injection technique before home use
4. Screen for priapism risk factors before initiating therapy
5. Consider intraurethral alprostadil (MUSE) for patients averse to injections

### Limitations

Long-term safety data beyond 2 years is limited. Head-to-head comparisons with
newer therapies such as low-intensity shockwave therapy are lacking. Most trials
excluded patients with severe cardiovascular disease, limiting generalizability
to this population. The psychological burden of injection therapy may affect
real-world adherence compared to oral medications.

### References

1. Smith AB et al. (2019). Alprostadil mechanism of action in erectile tissue.
   J Urol. https://pubmed.ncbi.nlm.nih.gov/12345678/
2. Johnson CD et al. (2020). Meta-analysis of intracavernosal alprostadil efficacy.
   J Sex Med. https://pubmed.ncbi.nlm.nih.gov/23456789/
3. Martinez R et al. (2018). Alprostadil in PDE5 inhibitor non-responders.
   Int J Impot Res. https://pubmed.ncbi.nlm.nih.gov/34567890/
"""


def format_synthesis_prompt(
    query: str,
    evidence_summary: str,
    drug_candidates: list[str],
    key_findings: list[str],
    mechanism_score: int,
    clinical_score: int,
    confidence: float,
) -> str:
    """Format the user prompt for narrative synthesis.

    Args:
        query: Original research question
        evidence_summary: Formatted summary of evidence papers
        drug_candidates: List of identified drug/treatment candidates
        key_findings: List of key findings from assessment
        mechanism_score: Mechanism evidence score (0-10)
        clinical_score: Clinical evidence score (0-10)
        confidence: Overall confidence (0.0-1.0)

    Returns:
        Formatted user prompt for the synthesis LLM
    """
    candidates_str = ", ".join(drug_candidates) if drug_candidates else "None identified"
    if key_findings:
        findings_str = "\n".join(f"- {f}" for f in key_findings)
    else:
        findings_str = "No specific findings extracted"

    return f"""Synthesize a narrative research report for the following query.

## Research Question
{query}

## Evidence Summary
{evidence_summary}

## Identified Drug/Treatment Candidates
{candidates_str}

## Key Findings from Evidence Assessment
{findings_str}

## Assessment Scores
- Mechanism Score: {mechanism_score}/10
- Clinical Evidence Score: {clinical_score}/10
- Overall Confidence: {confidence:.0%}

## Instructions
Generate a NARRATIVE research report following the structure in your system prompt.
Write in prose paragraphs, NOT bullet points (except for Recommendations section).
ONLY cite papers mentioned in the Evidence Summary above - do NOT invent references.

{FEW_SHOT_EXAMPLE}
"""
