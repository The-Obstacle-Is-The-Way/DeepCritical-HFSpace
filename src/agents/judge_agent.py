"""Judge agent wrapper for Magentic integration."""

from collections.abc import AsyncIterable
from typing import Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)

from src.orchestrator import JudgeHandlerProtocol
from src.utils.models import Evidence, JudgeAssessment


class JudgeAgent(BaseAgent):  # type: ignore[misc]
    """Wraps JudgeHandler as an AgentProtocol for Magentic."""

    def __init__(
        self,
        judge_handler: JudgeHandlerProtocol,
        evidence_store: dict[str, list[Evidence]],
    ) -> None:
        super().__init__(
            name="JudgeAgent",
            description="Evaluates evidence quality and determines if sufficient for synthesis",
        )
        self._handler = judge_handler
        self._evidence_store = evidence_store  # Shared state for evidence

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Assess evidence quality."""
        # Extract original question from messages
        question = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER and msg.text:
                    question = msg.text
                    break
                elif isinstance(msg, str):
                    question = msg
                    break
        elif isinstance(messages, str):
            question = messages
        elif isinstance(messages, ChatMessage) and messages.text:
            question = messages.text

        # Get evidence from shared store
        evidence = self._evidence_store.get("current", [])

        # Assess
        assessment: JudgeAssessment = await self._handler.assess(question, evidence)

        # Format response
        response_text = f"""## Assessment

**Sufficient**: {assessment.sufficient}
**Confidence**: {assessment.confidence:.0%}
**Recommendation**: {assessment.recommendation}

### Scores
- Mechanism: {assessment.details.mechanism_score}/10
- Clinical: {assessment.details.clinical_evidence_score}/10

### Reasoning
{assessment.reasoning}
"""

        if assessment.next_search_queries:
            response_text += "\n### Next Queries\n" + "\n".join(
                f"- {q}" for q in assessment.next_search_queries
            )

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"judge-{assessment.recommendation}",
            additional_properties={"assessment": assessment.model_dump()},
        )

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper for judge."""
        result = await self.run(messages, thread=thread, **kwargs)
        yield AgentRunResponseUpdate(messages=result.messages, response_id=result.response_id)
