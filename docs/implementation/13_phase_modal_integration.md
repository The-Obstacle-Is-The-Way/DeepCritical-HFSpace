# Phase 13 Implementation Spec: Modal Pipeline Integration

**Goal**: Wire existing Modal code execution into the agent pipeline.
**Philosophy**: "Sandboxed execution makes AI-generated code trustworthy."
**Prerequisite**: Phase 12 complete (MCP server working)
**Priority**: P1 - HIGH VALUE ($2,500 Modal Innovation Award)
**Estimated Time**: 2-3 hours

---

## 1. Why Modal Integration?

### Current State Analysis

Mario already implemented `src/tools/code_execution.py`:

| Component | Status | Notes |
|-----------|--------|-------|
| `ModalCodeExecutor` class | Built | Executes Python in Modal sandbox |
| `SANDBOX_LIBRARIES` | Defined | pandas, numpy, scipy, etc. |
| `execute()` method | Implemented | Stdout/stderr capture |
| `execute_with_return()` | Implemented | Returns `result` variable |
| `AnalysisAgent` | Built | Uses Modal for statistical analysis |
| **Pipeline Integration** | **MISSING** | Not wired into main orchestrator |

### What's Missing

```text
Current Flow:
  User Query → Orchestrator → Search → Judge → [Report] → Done

With Modal:
  User Query → Orchestrator → Search → Judge → [Analysis*] → Report → Done
                                                    ↓
                                          Modal Sandbox Execution
```

*The AnalysisAgent exists but is NOT called by either orchestrator.

---

## 2. Critical Dependency Analysis

### The Problem (Senior Feedback)

```python
# src/agents/analysis_agent.py - Line 8
from agent_framework import (
    AgentRunResponse,
    BaseAgent,
    ...
)
```

```toml
# pyproject.toml - agent-framework is OPTIONAL
[project.optional-dependencies]
magentic = [
    "agent-framework-core",
]
```

**If we import `AnalysisAgent` in the simple orchestrator without the `magentic` extra installed, the app CRASHES on startup.**

### The SOLID Solution

**Single Responsibility Principle**: Decouple Modal execution logic from `agent_framework`.

```text
BEFORE (Coupled):
  AnalysisAgent (requires agent_framework)
       ↓
  ModalCodeExecutor

AFTER (Decoupled):
  StatisticalAnalyzer (no agent_framework dependency)  ← Simple mode uses this
       ↓
  ModalCodeExecutor
       ↑
  AnalysisAgent (wraps StatisticalAnalyzer)  ← Magentic mode uses this
```

**Key insight**: Create `src/services/statistical_analyzer.py` with ZERO agent_framework imports.

---

## 3. Prize Opportunity

### Modal Innovation Award: $2,500

**Judging Criteria**:
1. **Sandbox Isolation** - Code runs in container, not local
2. **Scientific Computing** - Real pandas/scipy analysis
3. **Safety** - Can't access local filesystem
4. **Speed** - Modal's fast cold starts

### What We Need to Show

```python
# LLM generates analysis code
code = """
import pandas as pd
import scipy.stats as stats

data = pd.DataFrame({
    'study': ['Study1', 'Study2', 'Study3'],
    'effect_size': [0.45, 0.52, 0.38],
    'sample_size': [120, 85, 200]
})

weighted_mean = (data['effect_size'] * data['sample_size']).sum() / data['sample_size'].sum()
t_stat, p_value = stats.ttest_1samp(data['effect_size'], 0)

print(f"Weighted Effect Size: {weighted_mean:.3f}")
print(f"P-value: {p_value:.4f}")

result = "SUPPORTED" if p_value < 0.05 else "INCONCLUSIVE"
"""

# Executed SAFELY in Modal sandbox
executor = get_code_executor()
output = executor.execute(code)  # Runs in isolated container!
```

---

## 4. Technical Specification

### 4.1 Dependencies

```toml
# pyproject.toml - NO CHANGES to dependencies
# StatisticalAnalyzer uses only:
#   - pydantic-ai (already in main deps)
#   - modal (already in main deps)
#   - src.tools.code_execution (no agent_framework)
```

### 4.2 Environment Variables

```bash
# .env
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

### 4.3 Integration Points

| Integration Point | File | Change Required |
|-------------------|------|-----------------|
| New Service | `src/services/statistical_analyzer.py` | CREATE (no agent_framework) |
| Simple Orchestrator | `src/orchestrator.py` | Use `StatisticalAnalyzer` |
| Config | `src/utils/config.py` | Add `enable_modal_analysis` setting |
| AnalysisAgent | `src/agents/analysis_agent.py` | Refactor to wrap `StatisticalAnalyzer` |
| MCP Tool | `src/mcp_tools.py` | Add `analyze_hypothesis` tool |

---

## 5. Implementation

### 5.1 Configuration Update (`src/utils/config.py`)

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Modal Configuration
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    enable_modal_analysis: bool = False  # Opt-in for hackathon demo

    @property
    def modal_available(self) -> bool:
        """Check if Modal credentials are configured."""
        return bool(self.modal_token_id and self.modal_token_secret)
```

### 5.2 StatisticalAnalyzer Service (`src/services/statistical_analyzer.py`)

**This is the key fix - NO agent_framework imports.**

```python
"""Statistical analysis service using Modal code execution.

This module provides Modal-based statistical analysis WITHOUT depending on
agent_framework. This allows it to be used in the simple orchestrator mode
without requiring the magentic optional dependency.

The AnalysisAgent (in src/agents/) wraps this service for magentic mode.
"""

import asyncio
import re
from functools import partial
from typing import Any

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

    verdict: str = Field(
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
        # Build analysis prompt
        evidence_summary = self._summarize_evidence(evidence[:10])
        hypothesis_text = ""
        if hypothesis:
            hypothesis_text = f"""
Hypothesis: {hypothesis.get('drug', 'Unknown')} → {hypothesis.get('target', '?')} → {hypothesis.get('pathway', '?')} → {hypothesis.get('effect', '?')}
Confidence: {hypothesis.get('confidence', 0.5):.0%}
"""

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
                    statistical_evidence=f"Execution failed: {execution['error']}",
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
            lines.append(f"{i}. {ev.content[:200]}...")
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
        verdict = "INCONCLUSIVE"
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


# Singleton for reuse
_analyzer: StatisticalAnalyzer | None = None


def get_statistical_analyzer() -> StatisticalAnalyzer:
    """Get or create singleton StatisticalAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = StatisticalAnalyzer()
    return _analyzer
```

### 5.3 Simple Orchestrator Update (`src/orchestrator.py`)

**Uses `StatisticalAnalyzer` directly - NO agent_framework import.**

```python
"""Main orchestrator with optional Modal analysis."""

from src.utils.config import settings

# ... existing imports ...


class Orchestrator:
    """Search-Judge-Analyze orchestration loop."""

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        judge_handler: JudgeHandlerProtocol,
        config: OrchestratorConfig | None = None,
        enable_analysis: bool = False,  # New parameter
    ) -> None:
        self.search = search_handler
        self.judge = judge_handler
        self.config = config or OrchestratorConfig()
        self.history: list[dict[str, Any]] = []
        self._enable_analysis = enable_analysis and settings.modal_available

        # Lazy-load analysis (NO agent_framework dependency!)
        self._analyzer: Any = None

    def _get_analyzer(self) -> Any:
        """Lazy initialization of StatisticalAnalyzer.

        Note: This imports from src.services, NOT src.agents,
        so it works without the magentic optional dependency.
        """
        if self._analyzer is None:
            from src.services.statistical_analyzer import get_statistical_analyzer

            self._analyzer = get_statistical_analyzer()
        return self._analyzer

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Main orchestration loop with optional Modal analysis."""
        # ... existing search/judge loop ...

        # After judge says "synthesize", optionally run analysis
        if self._enable_analysis and assessment.recommendation == "synthesize":
            yield AgentEvent(
                type="analyzing",
                message="Running statistical analysis in Modal sandbox...",
                data={},
                iteration=iteration,
            )

            try:
                analyzer = self._get_analyzer()

                # Run Modal analysis (no agent_framework needed!)
                analysis_result = await analyzer.analyze(
                    query=query,
                    evidence=all_evidence,
                    hypothesis=None,  # Could add hypothesis generation later
                )

                yield AgentEvent(
                    type="analysis_complete",
                    message=f"Analysis verdict: {analysis_result.verdict}",
                    data=analysis_result.model_dump(),
                    iteration=iteration,
                )

            except Exception as e:
                yield AgentEvent(
                    type="error",
                    message=f"Modal analysis failed: {e}",
                    data={"error": str(e)},
                    iteration=iteration,
                )

        # Continue to synthesis...
```

### 5.4 Refactor AnalysisAgent (`src/agents/analysis_agent.py`)

**Wrap `StatisticalAnalyzer` for magentic mode.**

```python
"""Analysis agent for statistical analysis using Modal code execution.

This agent wraps StatisticalAnalyzer for use in magentic multi-agent mode.
The core logic is in src/services/statistical_analyzer.py to avoid
coupling agent_framework to the simple orchestrator.
"""

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

from src.services.statistical_analyzer import (
    AnalysisResult,
    get_statistical_analyzer,
)
from src.utils.models import Evidence

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class AnalysisAgent(BaseAgent):  # type: ignore[misc]
    """Wraps StatisticalAnalyzer for magentic multi-agent mode."""

    def __init__(
        self,
        evidence_store: dict[str, Any],
        embedding_service: "EmbeddingService | None" = None,
    ) -> None:
        super().__init__(
            name="AnalysisAgent",
            description="Performs statistical analysis using Modal sandbox",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service
        self._analyzer = get_statistical_analyzer()

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Analyze evidence and return verdict."""
        query = self._extract_query(messages)
        hypotheses = self._evidence_store.get("hypotheses", [])
        evidence = self._evidence_store.get("current", [])

        if not evidence:
            return self._error_response("No evidence available.")

        # Get primary hypothesis if available
        hypothesis_dict = None
        if hypotheses:
            h = hypotheses[0]
            hypothesis_dict = {
                "drug": getattr(h, "drug", "Unknown"),
                "target": getattr(h, "target", "?"),
                "pathway": getattr(h, "pathway", "?"),
                "effect": getattr(h, "effect", "?"),
                "confidence": getattr(h, "confidence", 0.5),
            }

        # Delegate to StatisticalAnalyzer
        result = await self._analyzer.analyze(
            query=query,
            evidence=evidence,
            hypothesis=hypothesis_dict,
        )

        # Store in shared context
        self._evidence_store["analysis"] = result.model_dump()

        # Format response
        response_text = self._format_response(result)

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"analysis-{result.verdict.lower()}",
            additional_properties={"analysis": result.model_dump()},
        )

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

        lines.extend([
            "\n### Statistical Evidence",
            "```",
            result.statistical_evidence,
            "```",
        ])
        return "\n".join(lines)

    def _error_response(self, message: str) -> AgentRunResponse:
        """Create error response."""
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=f"**Error**: {message}")],
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
```

### 5.5 MCP Tool for Modal Analysis (`src/mcp_tools.py`)

Add to existing MCP tools:

```python
async def analyze_hypothesis(
    drug: str,
    condition: str,
    evidence_summary: str,
) -> str:
    """Perform statistical analysis of drug repurposing hypothesis using Modal.

    Executes AI-generated Python code in a secure Modal sandbox to analyze
    the statistical evidence for a drug repurposing hypothesis.

    Args:
        drug: The drug being evaluated (e.g., "metformin")
        condition: The target condition (e.g., "Alzheimer's disease")
        evidence_summary: Summary of evidence to analyze

    Returns:
        Analysis result with verdict (SUPPORTED/REFUTED/INCONCLUSIVE) and statistics
    """
    from src.services.statistical_analyzer import get_statistical_analyzer
    from src.utils.config import settings
    from src.utils.models import Citation, Evidence

    if not settings.modal_available:
        return "Error: Modal credentials not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."

    # Create evidence from summary
    evidence = [
        Evidence(
            content=evidence_summary,
            citation=Citation(
                source="pubmed",
                title=f"Evidence for {drug} in {condition}",
                url="https://example.com",
                date="2024-01-01",
                authors=["User Provided"],
            ),
            relevance=0.9,
        )
    ]

    analyzer = get_statistical_analyzer()
    result = await analyzer.analyze(
        query=f"Can {drug} treat {condition}?",
        evidence=evidence,
        hypothesis={"drug": drug, "target": "unknown", "pathway": "unknown", "effect": condition},
    )

    return f"""## Statistical Analysis: {drug} for {condition}

### Verdict: **{result.verdict}**
**Confidence**: {result.confidence:.0%}

### Key Findings
{chr(10).join(f"- {f}" for f in result.key_findings) or "- No specific findings extracted"}

### Execution Output
```
{result.execution_output}
```

### Generated Code
```python
{result.code_generated}
```

**Executed in Modal Sandbox** - Isolated, secure, reproducible.
"""
```

### 5.6 Demo Scripts

#### `examples/modal_demo/verify_sandbox.py`

```python
#!/usr/bin/env python3
"""Verify that Modal sandbox is properly isolated.

This script proves to judges that code runs in Modal, not locally.
NO agent_framework dependency - uses only src.tools.code_execution.

Usage:
    uv run python examples/modal_demo/verify_sandbox.py
"""

import asyncio
from functools import partial

from src.tools.code_execution import get_code_executor
from src.utils.config import settings


async def main() -> None:
    """Verify Modal sandbox isolation."""
    if not settings.modal_available:
        print("Error: Modal credentials not configured.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env")
        return

    executor = get_code_executor()
    loop = asyncio.get_running_loop()

    print("=" * 60)
    print("Modal Sandbox Isolation Verification")
    print("=" * 60 + "\n")

    # Test 1: Hostname
    print("Test 1: Check hostname (should NOT be your machine)")
    code1 = "import socket; print(f'Hostname: {socket.gethostname()}')"
    result1 = await loop.run_in_executor(None, partial(executor.execute, code1))
    print(f"  {result1['stdout'].strip()}\n")

    # Test 2: Scientific libraries
    print("Test 2: Verify scientific libraries")
    code2 = """
import pandas as pd
import numpy as np
import scipy
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
"""
    result2 = await loop.run_in_executor(None, partial(executor.execute, code2))
    print(f"  {result2['stdout'].strip()}\n")

    # Test 3: Network blocked
    print("Test 3: Verify network isolation")
    code3 = """
import urllib.request
try:
    urllib.request.urlopen("https://google.com", timeout=2)
    print("Network: ALLOWED (unexpected!)")
except Exception:
    print("Network: BLOCKED (as expected)")
"""
    result3 = await loop.run_in_executor(None, partial(executor.execute, code3))
    print(f"  {result3['stdout'].strip()}\n")

    # Test 4: Real statistics
    print("Test 4: Execute statistical analysis")
    code4 = """
import pandas as pd
import scipy.stats as stats

data = pd.DataFrame({'effect': [0.42, 0.38, 0.51]})
mean = data['effect'].mean()
t_stat, p_val = stats.ttest_1samp(data['effect'], 0)

print(f"Mean Effect: {mean:.3f}")
print(f"P-value: {p_val:.4f}")
print(f"Verdict: {'SUPPORTED' if p_val < 0.05 else 'INCONCLUSIVE'}")
"""
    result4 = await loop.run_in_executor(None, partial(executor.execute, code4))
    print(f"  {result4['stdout'].strip()}\n")

    print("=" * 60)
    print("All tests complete - Modal sandbox verified!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

#### `examples/modal_demo/run_analysis.py`

```python
#!/usr/bin/env python3
"""Demo: Modal-powered statistical analysis.

This script uses StatisticalAnalyzer directly (NO agent_framework dependency).

Usage:
    uv run python examples/modal_demo/run_analysis.py "metformin alzheimer"
"""

import argparse
import asyncio
import os
import sys

from src.services.statistical_analyzer import get_statistical_analyzer
from src.tools.pubmed import PubMedTool
from src.utils.config import settings


async def main() -> None:
    """Run the Modal analysis demo."""
    parser = argparse.ArgumentParser(description="Modal Analysis Demo")
    parser.add_argument("query", help="Research query")
    args = parser.parse_args()

    if not settings.modal_available:
        print("Error: Modal credentials not configured.")
        sys.exit(1)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: No LLM API key found.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("DeepBoner Modal Analysis Demo")
    print(f"Query: {args.query}")
    print(f"{'=' * 60}\n")

    # Step 1: Gather Evidence
    print("Step 1: Gathering evidence from PubMed...")
    pubmed = PubMedTool()
    evidence = await pubmed.search(args.query, max_results=5)
    print(f"  Found {len(evidence)} papers\n")

    # Step 2: Run Modal Analysis
    print("Step 2: Running statistical analysis in Modal sandbox...")
    analyzer = get_statistical_analyzer()
    result = await analyzer.analyze(query=args.query, evidence=evidence)

    # Step 3: Display Results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nVerdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.0%}")
    print("\nKey Findings:")
    for finding in result.key_findings:
        print(f"  - {finding}")

    print("\n[Demo Complete - Code executed in Modal, not locally]")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. TDD Test Suite

### 6.1 Unit Tests (`tests/unit/services/test_statistical_analyzer.py`)

```python
"""Unit tests for StatisticalAnalyzer service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.statistical_analyzer import (
    AnalysisResult,
    StatisticalAnalyzer,
    get_statistical_analyzer,
)
from src.utils.models import Citation, Evidence


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Sample evidence for testing."""
    return [
        Evidence(
            content="Metformin shows effect size of 0.45.",
            citation=Citation(
                source="pubmed",
                title="Metformin Study",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2024-01-15",
                authors=["Smith J"],
            ),
            relevance=0.9,
        )
    ]


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer (no agent_framework dependency)."""

    def test_no_agent_framework_import(self) -> None:
        """StatisticalAnalyzer must NOT import agent_framework."""
        import src.services.statistical_analyzer as module

        # Check module doesn't import agent_framework
        source = open(module.__file__).read()
        assert "agent_framework" not in source
        assert "BaseAgent" not in source

    @pytest.mark.asyncio
    async def test_analyze_returns_result(
        self, sample_evidence: list[Evidence]
    ) -> None:
        """analyze() should return AnalysisResult."""
        analyzer = StatisticalAnalyzer()

        with patch.object(analyzer, "_get_agent") as mock_agent, \
             patch.object(analyzer, "_get_code_executor") as mock_executor:

            # Mock LLM
            mock_agent.return_value.run = AsyncMock(
                return_value=MagicMock(output="print('SUPPORTED')")
            )

            # Mock Modal
            mock_executor.return_value.execute.return_value = {
                "stdout": "SUPPORTED\np-value: 0.01",
                "stderr": "",
                "success": True,
            }

            result = await analyzer.analyze("test query", sample_evidence)

            assert isinstance(result, AnalysisResult)
            assert result.verdict == "SUPPORTED"

    def test_singleton(self) -> None:
        """get_statistical_analyzer should return singleton."""
        a1 = get_statistical_analyzer()
        a2 = get_statistical_analyzer()
        assert a1 is a2


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_verdict_values(self) -> None:
        """Verdict should be one of the expected values."""
        for verdict in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]:
            result = AnalysisResult(
                verdict=verdict,
                confidence=0.8,
                statistical_evidence="test",
                code_generated="print('test')",
                execution_output="test",
            )
            assert result.verdict == verdict

    def test_confidence_bounds(self) -> None:
        """Confidence must be 0.0-1.0."""
        with pytest.raises(ValueError):
            AnalysisResult(
                verdict="SUPPORTED",
                confidence=1.5,  # Invalid
                statistical_evidence="test",
                code_generated="test",
                execution_output="test",
            )
```

### 6.2 Integration Test (`tests/integration/test_modal.py`)

```python
"""Integration tests for Modal (requires credentials)."""

import pytest

from src.utils.config import settings


@pytest.mark.integration
@pytest.mark.skipif(not settings.modal_available, reason="Modal not configured")
class TestModalIntegration:
    """Integration tests requiring Modal credentials."""

    @pytest.mark.asyncio
    async def test_sandbox_executes_code(self) -> None:
        """Modal sandbox should execute Python code."""
        import asyncio
        from functools import partial

        from src.tools.code_execution import get_code_executor

        executor = get_code_executor()
        code = "import pandas as pd; print(pd.DataFrame({'a': [1,2,3]})['a'].sum())"

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, partial(executor.execute, code, timeout=30)
        )

        assert result["success"]
        assert "6" in result["stdout"]

    @pytest.mark.asyncio
    async def test_statistical_analyzer_works(self) -> None:
        """StatisticalAnalyzer should work end-to-end."""
        from src.services.statistical_analyzer import get_statistical_analyzer
        from src.utils.models import Citation, Evidence

        evidence = [
            Evidence(
                content="Drug shows 40% improvement in trial.",
                citation=Citation(
                    source="pubmed",
                    title="Test",
                    url="https://test.com",
                    date="2024-01-01",
                    authors=["Test"],
                ),
                relevance=0.9,
            )
        ]

        analyzer = get_statistical_analyzer()
        result = await analyzer.analyze("test drug efficacy", evidence)

        assert result.verdict in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]
        assert 0.0 <= result.confidence <= 1.0
```

---

## 7. Verification Commands

```bash
# 1. Verify NO agent_framework in StatisticalAnalyzer
grep -r "agent_framework" src/services/statistical_analyzer.py
# Should return nothing!

# 2. Run unit tests (no Modal needed)
uv run pytest tests/unit/services/test_statistical_analyzer.py -v

# 3. Run verification script (requires Modal)
uv run python examples/modal_demo/verify_sandbox.py

# 4. Run analysis demo (requires Modal + LLM)
uv run python examples/modal_demo/run_analysis.py "metformin alzheimer"

# 5. Run integration tests
uv run pytest tests/integration/test_modal.py -v -m integration

# 6. Full test suite
make check
```

---

## 8. Definition of Done

Phase 13 is **COMPLETE** when:

- [ ] `src/services/statistical_analyzer.py` created (NO agent_framework)
- [ ] `src/utils/config.py` has `enable_modal_analysis` setting
- [ ] `src/orchestrator.py` uses `StatisticalAnalyzer` directly
- [ ] `src/agents/analysis_agent.py` refactored to wrap `StatisticalAnalyzer`
- [ ] `src/mcp_tools.py` has `analyze_hypothesis` tool
- [ ] `examples/modal_demo/verify_sandbox.py` working
- [ ] `examples/modal_demo/run_analysis.py` working
- [ ] Unit tests pass WITHOUT magentic extra installed
- [ ] Integration tests pass WITH Modal credentials
- [ ] All lints pass

---

## 9. Architecture After Phase 13

```text
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Clients                              │
│              (Claude Desktop, Cursor, etc.)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP Protocol
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Gradio App + MCP Server                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MCP Tools: search_pubmed, search_trials, search_biorxiv │   │
│  │             search_all, analyze_hypothesis               │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────────────┐            ┌───────────────────────────┐
│   Simple Orchestrator │            │   Magentic Orchestrator   │
│  (no agent_framework) │            │   (with agent_framework)  │
│                       │            │                           │
│  SearchHandler        │            │  SearchAgent              │
│  JudgeHandler         │            │  JudgeAgent               │
│  StatisticalAnalyzer ─┼────────────┼→ AnalysisAgent ───────────┤
│                       │            │  (wraps StatisticalAnalyzer)
└───────────┬───────────┘            └───────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    StatisticalAnalyzer                           │
│              (src/services/statistical_analyzer.py)              │
│                    NO agent_framework dependency                 │
│                                                                  │
│  1. Generate code with pydantic-ai                               │
│  2. Execute in Modal sandbox                                     │
│  3. Return AnalysisResult                                        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Modal Sandbox                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - pandas, numpy, scipy, sklearn, statsmodels           │    │
│  │  - Network: BLOCKED                                     │    │
│  │  - Filesystem: ISOLATED                                 │    │
│  │  - Timeout: ENFORCED                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**This is the dependency-safe Modal stack.**

---

## 10. Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/services/statistical_analyzer.py` | **CREATE** | Core analysis (no agent_framework) |
| `src/utils/config.py` | MODIFY | Add `enable_modal_analysis` |
| `src/orchestrator.py` | MODIFY | Use `StatisticalAnalyzer` |
| `src/agents/analysis_agent.py` | MODIFY | Wrap `StatisticalAnalyzer` |
| `src/mcp_tools.py` | MODIFY | Add `analyze_hypothesis` |
| `examples/modal_demo/verify_sandbox.py` | CREATE | Sandbox verification |
| `examples/modal_demo/run_analysis.py` | CREATE | Demo script |
| `tests/unit/services/test_statistical_analyzer.py` | CREATE | Unit tests |
| `tests/integration/test_modal.py` | CREATE | Integration tests |

**Key Fix**: `StatisticalAnalyzer` has ZERO agent_framework imports, making it safe for the simple orchestrator.
