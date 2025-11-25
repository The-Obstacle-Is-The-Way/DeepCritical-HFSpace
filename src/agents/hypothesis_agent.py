"""Hypothesis agent for mechanistic reasoning."""

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
from src.prompts.hypothesis import SYSTEM_PROMPT, format_hypothesis_prompt
from src.utils.models import HypothesisAssessment

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class HypothesisAgent(BaseAgent):  # type: ignore[misc]
    """Generates mechanistic hypotheses based on evidence."""

    def __init__(
        self,
        evidence_store: dict[str, Any],
        embedding_service: "EmbeddingService | None" = None,  # NEW: for diverse selection
    ) -> None:
        super().__init__(
            name="HypothesisAgent",
            description="Generates scientific hypotheses about drug mechanisms to guide research",
        )
        self._evidence_store = evidence_store
        self._embeddings = embedding_service  # Used for MMR evidence selection
        self._agent: Agent[None, HypothesisAssessment] | None = None  # Lazy init

    def _get_agent(self) -> Agent[None, HypothesisAssessment]:
        """Lazy initialization of LLM agent to avoid requiring API keys at import."""
        if self._agent is None:
            self._agent = Agent(
                model=get_model(),  # Uses configured LLM (OpenAI/Anthropic)
                output_type=HypothesisAssessment,
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
        """Generate hypotheses based on current evidence."""
        # Extract query
        query = self._extract_query(messages)

        # Get current evidence
        evidence = self._evidence_store.get("current", [])

        if not evidence:
            return AgentRunResponse(
                messages=[
                    ChatMessage(
                        role=Role.ASSISTANT,
                        text="No evidence available yet. Search for evidence first.",
                    )
                ],
                response_id="hypothesis-no-evidence",
            )

        # Generate hypotheses with diverse evidence selection
        prompt = await format_hypothesis_prompt(query, evidence, embeddings=self._embeddings)
        result = await self._get_agent().run(prompt)
        assessment = result.output  # pydantic-ai returns .output for structured output

        # Store hypotheses in shared context
        existing = self._evidence_store.get("hypotheses", [])
        self._evidence_store["hypotheses"] = existing + assessment.hypotheses

        # Format response
        response_text = self._format_response(assessment)

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"hypothesis-{len(assessment.hypotheses)}",
            additional_properties={"assessment": assessment.model_dump()},
        )

    def _format_response(self, assessment: HypothesisAssessment) -> str:
        """Format hypothesis assessment as markdown."""
        lines = ["## Generated Hypotheses\n"]

        for i, h in enumerate(assessment.hypotheses, 1):
            lines.append(f"### Hypothesis {i} (Confidence: {h.confidence:.0%})")
            lines.append(f"**Mechanism**: {h.drug} -> {h.target} -> {h.pathway} -> {h.effect}")
            lines.append(f"**Suggested searches**: {', '.join(h.search_suggestions)}\n")

        if assessment.primary_hypothesis:
            lines.append("### Primary Hypothesis")
            h = assessment.primary_hypothesis
            lines.append(f"{h.drug} -> {h.target} -> {h.pathway} -> {h.effect}\n")

        if assessment.knowledge_gaps:
            lines.append("### Knowledge Gaps")
            for gap in assessment.knowledge_gaps:
                lines.append(f"- {gap}")

        if assessment.recommended_searches:
            lines.append("\n### Recommended Next Searches")
            for search in assessment.recommended_searches:
                lines.append(f"- `{search}`")

        return "\n".join(lines)

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
