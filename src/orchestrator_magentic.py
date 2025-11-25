"""Magentic-based orchestrator for DeepCritical.

NOTE: Magentic mode currently requires OpenAI API keys. The MagenticBuilder's
standard manager uses OpenAIChatClient. Anthropic support may be added when
the agent_framework provides an AnthropicChatClient.
"""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService

from agent_framework import (
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticOrchestratorMessageEvent,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.search_agent import SearchAgent
from src.orchestrator import JudgeHandlerProtocol, SearchHandlerProtocol
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError
from src.utils.models import AgentEvent, Evidence

logger = structlog.get_logger()


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis only if needed."""
    return f"{text[:max_len]}..." if len(text) > max_len else text


class MagenticOrchestrator:
    """
    Magentic-based orchestrator - same API as Orchestrator.

    Uses Microsoft Agent Framework's MagenticBuilder for multi-agent coordination.

    Note:
        Magentic mode requires OPENAI_API_KEY. The MagenticBuilder's standard
        manager currently only supports OpenAI. If you have only an Anthropic
        key, use the "simple" orchestrator mode instead.
    """

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        judge_handler: JudgeHandlerProtocol,
        max_rounds: int = 10,
    ) -> None:
        self._search_handler = search_handler
        self._judge_handler = judge_handler
        self._max_rounds = max_rounds
        self._evidence_store: dict[str, list[Evidence]] = {"current": []}

    def _init_embedding_service(self) -> "EmbeddingService | None":
        """Initialize embedding service if available."""
        try:
            from src.services.embeddings import get_embedding_service

            service = get_embedding_service()
            logger.info("Embedding service enabled")
            return service
        except ImportError:
            logger.info("Embedding service not available (dependencies missing)")
        except Exception as e:
            logger.warning("Failed to initialize embedding service", error=str(e))
        return None

    def _build_workflow(
        self,
        search_agent: SearchAgent,
        hypothesis_agent: HypothesisAgent,
        judge_agent: JudgeAgent,
    ) -> Any:
        """Build the Magentic workflow with participants."""
        if not settings.openai_api_key:
            raise ConfigurationError(
                "Magentic mode requires OPENAI_API_KEY. "
                "Set the key or use mode='simple' with Anthropic."
            )

        return (
            MagenticBuilder()
            .participants(
                searcher=search_agent,
                hypothesizer=hypothesis_agent,
                judge=judge_agent,
            )
            .with_standard_manager(
                chat_client=OpenAIChatClient(
                    model_id=settings.openai_model, api_key=settings.openai_api_key
                ),
                max_round_count=self._max_rounds,
                max_stall_count=3,
                max_reset_count=2,
            )
            .build()
        )

    def _format_task(self, query: str, has_embeddings: bool) -> str:
        """Format the task instruction for the manager."""
        semantic_note = ""
        if has_embeddings:
            semantic_note = """
The system has semantic search enabled. When evidence is found:
1. Related concepts will be automatically surfaced
2. Duplicates are removed by meaning, not just URL
3. Use the surfaced related concepts to refine searches
"""
        return f"""Research drug repurposing opportunities for: {query}
{semantic_note}
Workflow:
1. SearcherAgent: Find initial evidence from PubMed and web. SEND ONLY A SIMPLE KEYWORD QUERY.
2. HypothesisAgent: Generate mechanistic hypotheses (Drug -> Target -> Pathway -> Effect).
3. SearcherAgent: Use hypothesis-suggested queries for targeted search.
4. JudgeAgent: Evaluate if evidence supports hypotheses.
5. Repeat until confident or max rounds.

Focus on:
- Identifying specific molecular targets
- Understanding mechanism of action
- Finding supporting/contradicting evidence for hypotheses
"""

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the Magentic workflow - same API as simple Orchestrator.

        Yields AgentEvent objects for real-time UI updates.
        """
        logger.info("Starting Magentic orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research (Magentic mode): {query}",
            iteration=0,
        )

        # Initialize services and agents
        embedding_service = self._init_embedding_service()
        search_agent = SearchAgent(
            self._search_handler, self._evidence_store, embedding_service=embedding_service
        )
        judge_agent = JudgeAgent(self._judge_handler, self._evidence_store)
        hypothesis_agent = HypothesisAgent(
            self._evidence_store, embedding_service=embedding_service
        )

        # Build workflow and task
        workflow = self._build_workflow(search_agent, hypothesis_agent, judge_agent)
        task = self._format_task(query, embedding_service is not None)

        iteration = 0
        try:
            async for event in workflow.run_stream(task):
                agent_event = self._process_event(event, iteration)
                if agent_event:
                    if isinstance(event, MagenticAgentMessageEvent):
                        iteration += 1
                    yield agent_event
        except Exception as e:
            logger.error("Magentic workflow failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Workflow error: {e!s}",
                iteration=iteration,
            )

    def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
        """Process a workflow event and return an AgentEvent if applicable."""
        if isinstance(event, MagenticOrchestratorMessageEvent):
            message_text = (
                event.message.text if event.message and hasattr(event.message, "text") else ""
            )
            kind = getattr(event, "kind", "manager")
            if message_text:
                return AgentEvent(
                    type="judging",
                    message=f"Manager ({kind}): {_truncate(message_text)}",
                    iteration=iteration,
                )

        elif isinstance(event, MagenticAgentMessageEvent):
            agent_name = event.agent_id or "unknown"
            msg_text = (
                event.message.text if event.message and hasattr(event.message, "text") else ""
            )
            return self._agent_message_event(agent_name, msg_text, iteration + 1)

        elif isinstance(event, MagenticFinalResultEvent):
            final_text = (
                event.message.text
                if event.message and hasattr(event.message, "text")
                else "No result"
            )
            return AgentEvent(
                type="complete",
                message=final_text,
                data={"iterations": iteration},
                iteration=iteration,
            )

        elif isinstance(event, MagenticAgentDeltaEvent):
            if event.text:
                return AgentEvent(
                    type="streaming",
                    message=event.text,
                    data={"agent_id": event.agent_id},
                    iteration=iteration,
                )

        elif isinstance(event, WorkflowOutputEvent):
            if event.data:
                return AgentEvent(
                    type="complete",
                    message=str(event.data),
                    iteration=iteration,
                )

        return None

    def _agent_message_event(self, agent_name: str, msg_text: str, iteration: int) -> AgentEvent:
        """Create an AgentEvent for an agent message."""
        if "search" in agent_name.lower():
            return AgentEvent(
                type="search_complete",
                message=f"Search agent: {_truncate(msg_text)}",
                iteration=iteration,
            )
        elif "hypothes" in agent_name.lower():
            return AgentEvent(
                type="hypothesizing",
                message=f"Hypothesis agent: {_truncate(msg_text)}",
                iteration=iteration,
            )
        elif "judge" in agent_name.lower():
            return AgentEvent(
                type="judge_complete",
                message=f"Judge agent: {_truncate(msg_text)}",
                iteration=iteration,
            )
        return AgentEvent(
            type="judging",
            message=f"{agent_name}: {_truncate(msg_text)}",
            iteration=iteration,
        )
