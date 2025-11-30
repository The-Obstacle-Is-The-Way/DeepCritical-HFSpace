"""Advanced Orchestrator using Microsoft Agent Framework.

This orchestrator uses the ChatAgent pattern from Microsoft's agent-framework-core
package for multi-agent coordination. It provides richer orchestration capabilities
including specialized agents (Search, Hypothesis, Judge, Report) coordinated by
a manager agent.

Note: Previously named 'orchestrator_magentic.py' - renamed to eliminate confusion
with the 'magentic' PyPI package (which is a different library).

Design Patterns:
- Mediator: Manager agent coordinates between specialized agents
- Strategy: Different agents implement different strategies for their tasks
- Observer: Event stream allows UI to observe progress
"""

import asyncio
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
from src.agents.state import init_magentic_state
from src.orchestrators.base import OrchestratorProtocol
from src.utils.config import settings
from src.utils.llm_factory import check_magentic_requirements
from src.utils.models import AgentEvent
from src.utils.service_loader import get_embedding_service_if_available

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol

logger = structlog.get_logger()


class AdvancedOrchestrator(OrchestratorProtocol):
    """
    Advanced orchestrator using Microsoft Agent Framework ChatAgent pattern.

    Each agent has an internal LLM that understands natural language
    instructions from the manager and can call tools appropriately.

    This orchestrator provides:
    - Multi-agent coordination (Search, Hypothesis, Judge, Report)
    - Manager agent for workflow orchestration
    - Streaming events for real-time UI updates
    - Configurable timeouts and round limits
    """

    def __init__(
        self,
        max_rounds: int = 10,
        chat_client: OpenAIChatClient | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        """Initialize orchestrator.

        Args:
            max_rounds: Maximum coordination rounds
            chat_client: Optional shared chat client for agents
            api_key: Optional OpenAI API key (for BYOK)
            timeout_seconds: Maximum workflow duration (default: 10 minutes)
        """
        # Validate requirements only if no key provided
        if not chat_client and not api_key:
            check_magentic_requirements()

        self._max_rounds = max_rounds
        self._timeout_seconds = timeout_seconds
        self._chat_client: OpenAIChatClient | None

        if chat_client:
            self._chat_client = chat_client
        elif api_key:
            # Create client with user provided key
            self._chat_client = OpenAIChatClient(
                model_id=settings.openai_model,
                api_key=api_key,
            )
        else:
            # Fallback to env vars (will fail later if requirements check wasn't run/passed)
            self._chat_client = None

    def _init_embedding_service(self) -> "EmbeddingServiceProtocol | None":
        """Initialize embedding service if available."""
        return get_embedding_service_if_available()

    def _build_workflow(self) -> Any:
        """Build the workflow with ChatAgent participants."""
        # Create agents with internal LLMs
        search_agent = create_search_agent(self._chat_client)
        judge_agent = create_judge_agent(self._chat_client)
        hypothesis_agent = create_hypothesis_agent(self._chat_client)
        report_agent = create_report_agent(self._chat_client)

        # Manager chat client (orchestrates the agents)
        manager_client = self._chat_client or OpenAIChatClient(
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
        Run the workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        logger.info("Starting Advanced orchestrator", query=query)

        yield AgentEvent(
            type="started",
            message=f"Starting research (Advanced mode): {query}",
            iteration=0,
        )

        # Initialize context state
        embedding_service = self._init_embedding_service()
        init_magentic_state(query, embedding_service)

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

        # UX FIX: Yield thinking state before blocking workflow call
        # The workflow.run_stream() blocks for 2+ minutes on first LLM call
        yield AgentEvent(
            type="thinking",
            message=(
                "Multi-agent reasoning in progress... "
                "This may take 2-5 minutes for complex queries."
            ),
            iteration=0,
        )

        iteration = 0
        final_event_received = False

        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for event in workflow.run_stream(task):
                    agent_event = self._process_event(event, iteration)
                    if agent_event:
                        if isinstance(event, MagenticAgentMessageEvent):
                            iteration += 1
                            # Yield progress update before the agent action
                            yield AgentEvent(
                                type="progress",
                                message=f"Round {iteration}/{self._max_rounds}...",
                                iteration=iteration,
                            )

                        if agent_event.type == "complete":
                            final_event_received = True

                        yield agent_event

            # GUARANTEE: Always emit termination event if stream ends without one
            # (e.g., max rounds reached)
            if not final_event_received:
                logger.warning(
                    "Workflow ended without final event",
                    iterations=iteration,
                )
                yield AgentEvent(
                    type="complete",
                    message=(
                        f"Research completed after {iteration} agent rounds. "
                        "Max iterations reached - results may be partial. "
                        "Try a more specific query for better results."
                    ),
                    data={"iterations": iteration, "reason": "max_rounds_reached"},
                    iteration=iteration,
                )

        except TimeoutError:
            logger.warning("Workflow timed out", iterations=iteration)
            yield AgentEvent(
                type="complete",
                message="Research timed out. Synthesizing available evidence...",
                data={"reason": "timeout", "iterations": iteration},
                iteration=iteration,
            )

        except Exception as e:
            logger.error("Workflow failed", error=str(e))
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

    def _get_event_type_for_agent(self, agent_name: str) -> str:
        """Map agent name to appropriate event type.

        Args:
            agent_name: The agent ID from the workflow event

        Returns:
            Event type string matching AgentEvent.type Literal
        """
        agent_lower = agent_name.lower()
        if "search" in agent_lower:
            return "search_complete"
        if "judge" in agent_lower:
            return "judge_complete"
        if "hypothes" in agent_lower:
            return "hypothesizing"
        if "report" in agent_lower:
            return "synthesizing"
        return "judging"  # Default for unknown agents

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
            event_type = self._get_event_type_for_agent(agent_name)

            # All returned types are valid AgentEvent.type literals
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


def _create_deprecated_alias() -> type["AdvancedOrchestrator"]:
    """Create a deprecated alias that warns on use."""
    import warnings

    class MagenticOrchestrator(AdvancedOrchestrator):
        """Deprecated alias for AdvancedOrchestrator.

        .. deprecated:: 0.1.0
            Use :class:`AdvancedOrchestrator` instead.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize deprecated MagenticOrchestrator (use AdvancedOrchestrator)."""
            warnings.warn(
                "MagenticOrchestrator is deprecated, use AdvancedOrchestrator instead. "
                "The name 'magentic' was confusing with the 'magentic' PyPI package.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    return MagenticOrchestrator


# Backwards compatibility alias with deprecation warning
MagenticOrchestrator = _create_deprecated_alias()
