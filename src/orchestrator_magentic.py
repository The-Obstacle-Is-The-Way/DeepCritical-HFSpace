"""Magentic-based orchestrator using ChatAgent pattern."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import structlog
from agent_framework import (
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticOrchestratorMessageEvent,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

from src.agents.magentic_agents import (
    create_hypothesis_agent,
    create_judge_agent,
    create_report_agent,
    create_search_agent,
)
from src.state import init_magentic_state
from src.utils.config import settings
from src.utils.llm_factory import check_magentic_requirements
from src.utils.models import AgentEvent

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService

logger = structlog.get_logger()


class MagenticOrchestrator:
    """
    Magentic-based orchestrator using ChatAgent pattern.

    Each agent has an internal LLM that understands natural language
    instructions from the manager and can call tools appropriately.
    """

    def __init__(
        self,
        max_rounds: int = 10,
        chat_client: OpenAIChatClient | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            max_rounds: Maximum coordination rounds
            chat_client: Optional shared chat client for agents
        """
        # Validate requirements via centralized factory
        check_magentic_requirements()

        self._max_rounds = max_rounds
        self._chat_client = chat_client

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

    def _build_workflow(self) -> Any:
        """Build the Magentic workflow with ChatAgent participants."""
        # Create agents with internal LLMs
        search_agent = create_search_agent(self._chat_client)
        judge_agent = create_judge_agent(self._chat_client)
        hypothesis_agent = create_hypothesis_agent(self._chat_client)
        report_agent = create_report_agent(self._chat_client)

        # Manager chat client (orchestrates the agents)
        manager_client = OpenAIChatClient(
            model_id=settings.openai_model,  # Use configured model
            api_key=settings.openai_api_key,
        )

        return (
            MagenticBuilder()
            .participants(
                searcher=search_agent,
                hypothesizer=hypothesis_agent,
                judge=judge_agent,
                reporter=report_agent,
            )
            .with_standard_manager(
                chat_client=manager_client,
                max_round_count=self._max_rounds,
                max_stall_count=3,
                max_reset_count=2,
            )
            .build()
        )

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the Magentic workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        logger.info("Starting Magentic orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research (Magentic mode): {query}",
            iteration=0,
        )

        # Initialize context state
        embedding_service = self._init_embedding_service()
        init_magentic_state(embedding_service)

        workflow = self._build_workflow()

        task = f"""Research drug repurposing opportunities for: {query}

Workflow:
1. SearchAgent: Find evidence from PubMed, ClinicalTrials.gov, and Europe PMC
2. HypothesisAgent: Generate mechanistic hypotheses (Drug -> Target -> Pathway -> Effect)
3. JudgeAgent: Evaluate if evidence is sufficient
4. If insufficient -> SearchAgent refines search based on gaps
5. If sufficient -> ReportAgent synthesizes final report

Focus on:
- Identifying specific molecular targets
- Understanding mechanism of action
- Finding clinical evidence supporting hypotheses

The final output should be a structured research report."""

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

    def _extract_text(self, message: Any) -> str:
        """
        Defensively extract text from a message object.

        Fixes bug where message.text might return the object itself or its repr.
        """
        if not message:
            return ""

        # Priority 1: .content (often the raw string or list of content)
        if hasattr(message, "content") and message.content:
            content = message.content
            # If it's a list (e.g., Multi-modal), join text parts
            if isinstance(content, list):
                return " ".join([str(c.text) for c in content if hasattr(c, "text")])
            return str(content)

        # Priority 2: .text (standard, but sometimes buggy/missing)
        if hasattr(message, "text") and message.text:
            # Verify it's not the object itself or a repr string
            text = str(message.text)
            if text.startswith("<") and "object at" in text:
                # Likely a repr string, ignore if possible
                pass
            else:
                return text

        # Fallback: If we can't find clean text, return str(message)
        # taking care to avoid infinite recursion if str() calls .text
        return str(message)

    def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
        """Process workflow event into AgentEvent."""
        if isinstance(event, MagenticOrchestratorMessageEvent):
            text = self._extract_text(event.message)
            if text:
                return AgentEvent(
                    type="judging",
                    message=f"Manager ({event.kind}): {text[:200]}...",
                    iteration=iteration,
                )

        elif isinstance(event, MagenticAgentMessageEvent):
            agent_name = event.agent_id or "unknown"
            text = self._extract_text(event.message)

            event_type = "judging"
            if "search" in agent_name.lower():
                event_type = "search_complete"
            elif "judge" in agent_name.lower():
                event_type = "judge_complete"
            elif "hypothes" in agent_name.lower():
                event_type = "hypothesizing"
            elif "report" in agent_name.lower():
                event_type = "synthesizing"

            return AgentEvent(
                type=event_type,  # type: ignore[arg-type]
                message=f"{agent_name}: {text[:200]}...",
                iteration=iteration + 1,
            )

        elif isinstance(event, MagenticFinalResultEvent):
            text = self._extract_text(event.message) if event.message else "No result"
            return AgentEvent(
                type="complete",
                message=text,
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
