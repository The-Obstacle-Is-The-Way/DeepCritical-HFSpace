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

        lines.extend(
            [
                "\n### Statistical Evidence",
                "```",
                result.statistical_evidence,
                "```",
            ]
        )
        return "\n".join(lines)

    def _error_response(self, message: str) -> AgentRunResponse:
        """Create error response."""
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=f"**Error**: {message}")],
            response_id="analysis-error",
        )

    def _extract_query(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None,
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
