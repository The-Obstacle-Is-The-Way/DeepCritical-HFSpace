"""Statistical analysis service using Modal code execution.

This module provides Modal-based statistical analysis WITHOUT depending on
agent_framework. This allows it to be used in the simple orchestrator mode
without requiring the magentic optional dependency.

The AnalysisAgent (in src/agents/) wraps this service for magentic mode.
"""

import asyncio
import re
from functools import lru_cache, partial
from typing import Any, Literal

# Type alias for verdict values
VerdictType = Literal["SUPPORTED", "REFUTED", "INCONCLUSIVE"]

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agent_factory.judges import get_model
from src.tools.code_execution import (
    CodeExecutionError,
    get_code_executor,
    get_sandbox_library_prompt,
)
from src.utils.models import Evidence


class AnalysisResult(BaseModel):
    """Result of statistical analysis."""

    verdict: VerdictType = Field(
        description="SUPPORTED, REFUTED, or INCONCLUSIVE",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in verdict (0-1)")
    statistical_evidence: str = Field(
        description="Summary of statistical findings from code execution"
    )
    code_generated: str = Field(description="Python code that was executed")
    execution_output: str = Field(description="Output from code execution")
    key_findings: list[str] = Field(default_factory=list, description="Key takeaways")
    limitations: list[str] = Field(default_factory=list, description="Limitations")


class StatisticalAnalyzer:
    """Performs statistical analysis using Modal code execution.

    This service:
    1. Generates Python code for statistical analysis using LLM
    2. Executes code in Modal sandbox
    3. Interprets results
    4. Returns verdict (SUPPORTED/REFUTED/INCONCLUSIVE)

    Note: This class has NO agent_framework dependency, making it safe
    to use in the simple orchestrator without the magentic extra.
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self._code_executor: Any = None
        self._agent: Agent[None, str] | None = None

    def _get_code_executor(self) -> Any:
        """Lazy initialization of code executor."""
        if self._code_executor is None:
            self._code_executor = get_code_executor()
        return self._code_executor

    def _get_agent(self) -> Agent[None, str]:
        """Lazy initialization of LLM agent for code generation."""
        if self._agent is None:
            library_versions = get_sandbox_library_prompt()
            self._agent = Agent(
                model=get_model(),
                output_type=str,
                system_prompt=f"""You are a biomedical data scientist.

Generate Python code to analyze research evidence and test hypotheses.

Guidelines:
1. Use pandas, numpy, scipy.stats for analysis
2. Print clear, interpretable results
3. Include statistical tests (t-tests, chi-square, etc.)
4. Calculate effect sizes and confidence intervals
5. Keep code concise (<50 lines)
6. Set 'result' variable to SUPPORTED, REFUTED, or INCONCLUSIVE

Available libraries:
{library_versions}

Output format: Return ONLY executable Python code, no explanations.""",
            )
        return self._agent

    async def analyze(
        self,
        query: str,
        evidence: list[Evidence],
        hypothesis: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Run statistical analysis on evidence.

        Args:
            query: The research question
            evidence: List of Evidence objects to analyze
            hypothesis: Optional hypothesis dict with drug, target, pathway, effect

        Returns:
            AnalysisResult with verdict and statistics
        """
        # Build analysis prompt (method handles slicing internally)
        evidence_summary = self._summarize_evidence(evidence)
        hypothesis_text = ""
        if hypothesis:
            hypothesis_text = (
                f"\nHypothesis: {hypothesis.get('drug', 'Unknown')} → "
                f"{hypothesis.get('target', '?')} → "
                f"{hypothesis.get('pathway', '?')} → "
                f"{hypothesis.get('effect', '?')}\n"
                f"Confidence: {hypothesis.get('confidence', 0.5):.0%}\n"
            )

        prompt = f"""Generate Python code to statistically analyze:

**Research Question**: {query}
{hypothesis_text}

**Evidence Summary**:
{evidence_summary}

Generate executable Python code to analyze this evidence."""

        try:
            # Generate code
            agent = self._get_agent()
            code_result = await agent.run(prompt)
            generated_code = code_result.output

            # Execute in Modal sandbox
            loop = asyncio.get_running_loop()
            executor = self._get_code_executor()
            execution = await loop.run_in_executor(
                None, partial(executor.execute, generated_code, timeout=120)
            )

            if not execution["success"]:
                return AnalysisResult(
                    verdict="INCONCLUSIVE",
                    confidence=0.0,
                    statistical_evidence=(
                        f"Execution failed: {execution.get('error', 'Unknown error')}"
                    ),
                    code_generated=generated_code,
                    execution_output=execution.get("stderr", ""),
                    key_findings=[],
                    limitations=["Code execution failed"],
                )

            # Interpret results
            return self._interpret_results(generated_code, execution)

        except CodeExecutionError as e:
            return AnalysisResult(
                verdict="INCONCLUSIVE",
                confidence=0.0,
                statistical_evidence=str(e),
                code_generated="",
                execution_output="",
                key_findings=[],
                limitations=[f"Analysis error: {e}"],
            )

    def _summarize_evidence(self, evidence: list[Evidence]) -> str:
        """Summarize evidence for code generation prompt."""
        if not evidence:
            return "No evidence available."

        lines = []
        for i, ev in enumerate(evidence[:5], 1):
            content = ev.content
            truncated = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"{i}. {truncated}")
            lines.append(f"   Source: {ev.citation.title}")
            lines.append(f"   Relevance: {ev.relevance:.0%}\n")

        return "\n".join(lines)

    def _interpret_results(
        self,
        code: str,
        execution: dict[str, Any],
    ) -> AnalysisResult:
        """Interpret code execution results."""
        stdout = execution["stdout"]
        stdout_upper = stdout.upper()

        # Extract verdict with robust word-boundary matching
        verdict: VerdictType = "INCONCLUSIVE"
        if re.search(r"\bSUPPORTED\b", stdout_upper) and not re.search(
            r"\b(?:NOT|UN)SUPPORTED\b", stdout_upper
        ):
            verdict = "SUPPORTED"
        elif re.search(r"\bREFUTED\b", stdout_upper):
            verdict = "REFUTED"

        # Extract key findings
        key_findings = []
        for line in stdout.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["p-value", "significant", "effect", "mean"]):
                key_findings.append(line.strip())

        # Calculate confidence from p-values
        confidence = self._calculate_confidence(stdout)

        return AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            statistical_evidence=stdout.strip(),
            code_generated=code,
            execution_output=stdout,
            key_findings=key_findings[:5],
            limitations=[
                "Analysis based on summary data only",
                "Limited to available evidence",
                "Statistical tests assume data independence",
            ],
        )

    def _calculate_confidence(self, output: str) -> float:
        """Calculate confidence based on statistical results."""
        p_values = re.findall(r"p[-\s]?value[:\s]+(\d+\.?\d*)", output.lower())

        if p_values:
            try:
                min_p = min(float(p) for p in p_values)
                if min_p < 0.001:
                    return 0.95
                elif min_p < 0.01:
                    return 0.90
                elif min_p < 0.05:
                    return 0.80
                else:
                    return 0.60
            except ValueError:
                pass

        return 0.70  # Default


@lru_cache(maxsize=1)
def get_statistical_analyzer() -> StatisticalAnalyzer:
    """Get or create singleton StatisticalAnalyzer instance (thread-safe via lru_cache)."""
    return StatisticalAnalyzer()
