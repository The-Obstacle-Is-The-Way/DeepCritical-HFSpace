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

from src.agent_factory.judges import get_model
from src.prompts.report import SYSTEM_PROMPT, format_report_prompt
from src.utils.citation_validator import validate_references
from src.utils.models import Evidence, ResearchReport

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class ReportAgent(BaseAgent):  # type: ignore[misc]
    """Generates structured scientific reports from evidence and hypotheses."""

    def __init__(
        self,
        evidence_store: dict[str, Any],
        embedding_service: "EmbeddingService | None" = None,  # For diverse selection
    ) -> None:
        super().__init__(
            name="ReportAgent",
            description="Generates structured scientific research reports with citations",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service
        self._agent: Agent[None, ResearchReport] | None = None  # Lazy init

    def _get_agent(self) -> Agent[None, ResearchReport]:
        """Lazy initialization of LLM agent to avoid requiring API keys at import."""
        if self._agent is None:
            self._agent = Agent(
                model=get_model(),
                output_type=ResearchReport,
                system_prompt=SYSTEM_PROMPT,
            )
        return self._agent

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
        evidence: list[Evidence] = self._evidence_store.get("current", [])
        hypotheses = self._evidence_store.get("hypotheses", [])
        assessment = self._evidence_store.get("last_assessment", {})

        if not evidence:
            return AgentRunResponse(
                messages=[
                    ChatMessage(
                        role=Role.ASSISTANT,
                        text="Cannot generate report: No evidence collected.",
                    )
                ],
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

        result = await self._get_agent().run(prompt)
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
