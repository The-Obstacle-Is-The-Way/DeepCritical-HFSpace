"""Analysis agent for statistical analysis using Modal code execution."""

import asyncio
from collections.abc import AsyncIterable
from functools import partial
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agent_factory.judges import get_model
from src.tools.code_execution import (
    CodeExecutionError,
    get_code_executor,
    get_sandbox_library_prompt,
)
from src.utils.models import Evidence

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class AnalysisResult(BaseModel):
    """Result of statistical analysis."""

    verdict: str = Field(
        description="SUPPORTED, REFUTED, or INCONCLUSIVE",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in verdict (0-1)")
    statistical_evidence: str = Field(
        description="Summary of statistical findings from code execution"
    )
    code_generated: str = Field(description="Python code that was executed")
    execution_output: str = Field(description="Output from code execution")
    key_findings: list[str] = Field(default_factory=list, description="Key takeaways from analysis")
    limitations: list[str] = Field(default_factory=list, description="Limitations of the analysis")


class AnalysisAgent(BaseAgent):  # type: ignore[misc]
    """Performs statistical analysis using Modal code execution.

    This agent:
    1. Retrieves relevant evidence using RAG (if available)
    2. Generates Python code for statistical analysis
    3. Executes code in Modal sandbox
    4. Interprets results
    5. Returns verdict (SUPPORTED/REFUTED/INCONCLUSIVE)
    """

    def __init__(
        self,
        evidence_store: dict[str, Any],
        embedding_service: "EmbeddingService | None" = None,
    ) -> None:
        super().__init__(
            name="AnalysisAgent",
            description="Performs statistical analysis of evidence using secure code execution",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service
        self._code_executor: Any = None  # Lazy initialized
        self._agent: Agent[None, str] | None = None  # LLM for code generation

    def _get_code_executor(self) -> Any:
        """Lazy initialization of code executor (avoids failing if Modal not configured)."""
        if self._code_executor is None:
            self._code_executor = get_code_executor()
        return self._code_executor

    def _get_agent(self) -> Agent[None, str]:
        """Lazy initialization of LLM agent."""
        if self._agent is None:
            self._agent = Agent(
                model=get_model(),
                output_type=str,  # Returns code as string
                system_prompt=self._get_system_prompt(),
            )
        return self._agent

    def _get_system_prompt(self) -> str:
        """System prompt for code generation."""
        library_versions = get_sandbox_library_prompt()
        return f"""You are a biomedical data scientist specializing in statistical analysis.

Your task: Generate Python code to analyze research evidence and test hypotheses.

Guidelines:
1. Use pandas, numpy, scipy.stats for analysis
2. Generate code that prints clear, interpretable results
3. Include statistical tests (t-tests, chi-square, meta-analysis, etc.)
4. Calculate effect sizes and confidence intervals
5. Print summary statistics and test results
6. Keep code concise (<50 lines)
7. Set a variable called 'result' with final verdict

Available libraries:
{library_versions}

Output format:
Return ONLY executable Python code, no explanations or markdown.
"""

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Analyze evidence and return verdict."""
        # Extract query and hypothesis
        query = self._extract_query(messages)
        hypotheses = self._evidence_store.get("hypotheses", [])
        evidence = self._evidence_store.get("current", [])

        if not hypotheses:
            return self._error_response("No hypotheses available. Run HypothesisAgent first.")

        if not evidence:
            return self._error_response("No evidence available. Run SearchAgent first.")

        # Get primary hypothesis (guaranteed to exist after check above)
        primary = hypotheses[0]

        # Retrieve relevant evidence using RAG (if available)
        relevant_evidence = await self._retrieve_relevant_evidence(primary, evidence)

        # Generate analysis code
        code_prompt = self._create_code_generation_prompt(query, primary, relevant_evidence)

        try:
            # Generate code using LLM
            agent = self._get_agent()
            code_result = await agent.run(code_prompt)
            generated_code = code_result.output

            # Execute code in Modal sandbox (run in thread to avoid blocking event loop)
            loop = asyncio.get_running_loop()
            executor = self._get_code_executor()
            execution_result = await loop.run_in_executor(
                None, partial(executor.execute, generated_code, timeout=120)
            )

            if not execution_result["success"]:
                return self._error_response(f"Code execution failed: {execution_result['error']}")

            # Interpret results
            analysis_result = await self._interpret_results(
                query, primary, generated_code, execution_result
            )

            # Store analysis in shared context
            self._evidence_store["analysis"] = analysis_result.model_dump()

            # Format response
            response_text = self._format_response(analysis_result)

            return AgentRunResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
                response_id=f"analysis-{analysis_result.verdict.lower()}",
                additional_properties={"analysis": analysis_result.model_dump()},
            )

        except CodeExecutionError as e:
            return self._error_response(f"Analysis failed: {e}")
        except Exception as e:
            return self._error_response(f"Unexpected error: {e}")

    async def _retrieve_relevant_evidence(
        self, hypothesis: Any, all_evidence: list[Evidence]
    ) -> list[Evidence]:
        """Retrieve most relevant evidence using RAG (if available).

        TODO: When embeddings service is available (self._embeddings),
        use semantic search to find evidence most relevant to the hypothesis.
        For now, returns top 10 evidence items.
        """
        # Future: Use self._embeddings for semantic search
        return all_evidence[:10]

    def _create_code_generation_prompt(
        self, query: str, hypothesis: Any, evidence: list[Evidence]
    ) -> str:
        """Create prompt for code generation."""
        # Extract data from evidence
        evidence_summary = self._summarize_evidence(evidence)

        prompt = f"""Generate Python code to statistically analyze the following hypothesis:

**Original Question**: {query}

**Hypothesis**: {hypothesis.drug} → {hypothesis.target} → {hypothesis.pathway} → {hypothesis.effect}
**Confidence**: {hypothesis.confidence:.0%}

**Evidence Summary**:
{evidence_summary}

**Task**:
1. Parse the evidence data
2. Perform appropriate statistical tests
3. Calculate effect sizes and confidence intervals
4. Determine verdict: SUPPORTED, REFUTED, or INCONCLUSIVE
5. Set result variable to verdict string

Generate executable Python code only (no markdown, no explanations).
"""
        return prompt

    def _summarize_evidence(self, evidence: list[Evidence]) -> str:
        """Summarize evidence for code generation prompt."""
        if not evidence:
            return "No evidence available."

        lines = []
        for i, ev in enumerate(evidence[:5], 1):  # Top 5 most relevant
            lines.append(f"{i}. {ev.content[:200]}...")
            lines.append(f"   Source: {ev.citation.title}")
            lines.append(f"   Relevance: {ev.relevance:.0%}\n")

        return "\n".join(lines)

    async def _interpret_results(
        self,
        query: str,
        hypothesis: Any,
        code: str,
        execution_result: dict[str, Any],
    ) -> AnalysisResult:
        """Interpret code execution results using LLM."""
        import re

        # Extract verdict from output using robust word-boundary matching
        stdout = execution_result["stdout"]
        stdout_upper = stdout.upper()
        verdict = "INCONCLUSIVE"  # Default

        # Avoid false positives like "NOT SUPPORTED" or "UNSUPPORTED"
        if re.search(r"\bSUPPORTED\b", stdout_upper) and not re.search(
            r"\b(?:NOT|UN)SUPPORTED\b", stdout_upper
        ):
            verdict = "SUPPORTED"
        elif re.search(r"\bREFUTED\b", stdout_upper):
            verdict = "REFUTED"
        elif re.search(r"\bINCONCLUSIVE\b", stdout_upper):
            verdict = "INCONCLUSIVE"

        # Parse key findings from output
        key_findings = self._extract_findings(stdout)

        # Calculate confidence based on statistical significance
        confidence = self._calculate_confidence(stdout)

        return AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            statistical_evidence=stdout.strip(),
            code_generated=code,
            execution_output=stdout,
            key_findings=key_findings,
            limitations=[
                "Analysis based on summary data only",
                "Limited to available evidence",
                "Statistical tests assume data independence",
            ],
        )

    def _extract_findings(self, output: str) -> list[str]:
        """Extract key findings from code output."""
        findings = []

        # Look for common statistical patterns
        lines = output.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(
                keyword in line_lower
                for keyword in ["p-value", "significant", "effect size", "correlation", "mean"]
            ):
                findings.append(line.strip())

        return findings[:5]  # Top 5 findings

    def _calculate_confidence(self, output: str) -> float:
        """Calculate confidence based on statistical results."""
        # Look for p-values
        import re

        p_values = re.findall(r"p[-\s]?value[:\s]+(\d+\.?\d*)", output.lower())

        if p_values:
            try:
                min_p = min(float(p) for p in p_values)
                # Higher confidence for lower p-values
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

        # Default medium confidence
        return 0.70

    def _format_response(self, result: AnalysisResult) -> str:
        """Format analysis result as markdown."""
        lines = [
            "## Statistical Analysis Complete\n",
            f"### Verdict: **{result.verdict}**",
            f"**Confidence**: {result.confidence:.0%}\n",
            "### Key Findings",
        ]

        for finding in result.key_findings:
            lines.append(f"- {finding}")

        lines.extend(
            [
                "\n### Statistical Evidence",
                "```",
                result.statistical_evidence,
                "```",
                "\n### Generated Code",
                "```python",
                result.code_generated,
                "```",
                "\n### Limitations",
            ]
        )

        for limitation in result.limitations:
            lines.append(f"- {limitation}")

        return "\n".join(lines)

    def _error_response(self, message: str) -> AgentRunResponse:
        """Create error response."""
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=f"❌ **Error**: {message}")],
            response_id="analysis-error",
        )

    def _extract_query(
        self, messages: str | ChatMessage | list[str] | list[ChatMessage] | None
    ) -> str:
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
        yield AgentRunResponseUpdate(messages=result.messages, response_id=result.response_id)
