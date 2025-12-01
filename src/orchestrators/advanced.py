"""
Advanced Orchestrator using Microsoft Agent Framework.

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

from src.agents.magentic_agents import (
    create_hypothesis_agent,
    create_judge_agent,
    create_report_agent,
    create_search_agent,
)
from src.agents.state import init_magentic_state
from src.clients.base import BaseChatClient
from src.clients.factory import get_chat_client
from src.config.domain import ResearchDomain, get_domain_config
from src.orchestrators.base import OrchestratorProtocol
from src.utils.config import settings
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

    # Estimated seconds per coordination round (for progress UI)
    _EST_SECONDS_PER_ROUND: int = 45

    def __init__(
        self,
        max_rounds: int = 5,
        chat_client: BaseChatClient | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        domain: ResearchDomain | str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Initialize the advanced orchestrator.

        Args:
            max_rounds: Maximum number of coordination rounds.
            chat_client: Optional pre-configured chat client.
            provider: Optional provider override ("openai", "huggingface").
            api_key: Optional API key override.
            domain: Research domain for customization.
            timeout_seconds: Optional timeout override (defaults to settings).
        """
        self._max_rounds = max_rounds
        self.domain = domain or ResearchDomain.SEXUAL_HEALTH
        self.domain_config = get_domain_config(self.domain)
        self._timeout_seconds = timeout_seconds or settings.advanced_timeout

        self.logger = logger.bind(orchestrator="advanced")

        # Use provided client or create one via factory
        self._chat_client = chat_client or get_chat_client(
            provider=provider,
            api_key=api_key,
        )

        # Event stream for UI updates
        self._events: list[AgentEvent] = []

        # Initialize services lazily
        self._embedding_service: EmbeddingServiceProtocol | None = None

        # Track execution statistics
        self.stats = {
            "rounds": 0,
            "searches": 0,
            "hypotheses": 0,
            "reports": 0,
            "errors": 0,
        }

    def _init_embedding_service(self) -> "EmbeddingServiceProtocol | None":
        """Initialize embedding service if available."""
        return get_embedding_service_if_available()

    def _build_workflow(self) -> Any:
        """Build the workflow with ChatAgent participants."""
        # Create agents with internal LLMs
        search_agent = create_search_agent(self._chat_client, domain=self.domain)
        judge_agent = create_judge_agent(self._chat_client, domain=self.domain)
        hypothesis_agent = create_hypothesis_agent(self._chat_client, domain=self.domain)
        report_agent = create_report_agent(self._chat_client, domain=self.domain)

        # Manager chat client (orchestrates the agents)
        manager_client = self._chat_client

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

    def _create_task_prompt(self, query: str) -> str:
        """Create the initial task prompt for the manager agent."""
        return f"""Research {self.domain_config.report_focus} for: {query}

## CRITICAL RULE
When JudgeAgent says "SUFFICIENT EVIDENCE" or "STOP SEARCHING":
→ IMMEDIATELY delegate to ReportAgent for synthesis
→ Do NOT continue searching or gathering more evidence
→ The Judge has determined evidence quality is adequate

## Standard Workflow
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

    def _get_progress_message(self, iteration: int) -> str:
        """Generate progress message with time estimation."""
        rounds_remaining = max(self._max_rounds - iteration, 0)
        est_seconds = rounds_remaining * self._EST_SECONDS_PER_ROUND
        if est_seconds >= 60:
            est_display = f"{est_seconds // 60}m {est_seconds % 60}s"
        else:
            est_display = f"{est_seconds}s"

        return f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)"

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
        yield AgentEvent(
            type="progress",
            message="Loading embedding service (LlamaIndex/ChromaDB)...",
            iteration=0,
        )
        embedding_service = self._init_embedding_service()

        yield AgentEvent(
            type="progress",
            message="Initializing research memory...",
            iteration=0,
        )
        init_magentic_state(query, embedding_service)

        yield AgentEvent(
            type="progress",
            message="Building agent team (Search, Judge, Hypothesis, Report)...",
            iteration=0,
        )
        workflow = self._build_workflow()

        task = self._create_task_prompt(query)

        # UX FIX: Yield thinking state before blocking workflow call
        # The workflow.run_stream() blocks for 2+ minutes on first LLM call
        yield AgentEvent(
            type="thinking",
            message=(
                f"Multi-agent reasoning in progress ({self._max_rounds} rounds max)... "
                f"Estimated time: {self._max_rounds * 45 // 60}-"
                f"{self._max_rounds * 60 // 60} minutes."
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
                            progress_msg = self._get_progress_message(iteration)

                            # Yield progress update before the agent action
                            yield AgentEvent(
                                type="progress",
                                message=progress_msg,
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

            # ACTUALLY synthesize from gathered evidence
            try:
                from src.agents.magentic_agents import create_report_agent
                from src.agents.state import get_magentic_state

                state = get_magentic_state()
                memory = state.memory

                # Get evidence summary from memory
                evidence_summary = await memory.get_context_summary()

                # Create and invoke ReportAgent for synthesis
                report_agent = create_report_agent(self._chat_client, domain=self.domain)

                yield AgentEvent(
                    type="synthesizing",
                    message="Workflow timed out. Synthesizing available evidence...",
                    iteration=iteration,
                )

                # Invoke ReportAgent directly
                # Note: ChatAgent.run() returns the final response string
                synthesis_result = await report_agent.run(
                    "Synthesize research report from this evidence. "
                    f"If evidence is sparse, say so.\n\n{evidence_summary}"
                )

                yield AgentEvent(
                    type="complete",
                    message=str(synthesis_result),
                    data={"reason": "timeout_synthesis", "iterations": iteration},
                    iteration=iteration,
                )
            except Exception as synth_error:
                logger.error("Timeout synthesis failed", error=str(synth_error))
                yield AgentEvent(
                    type="complete",
                    message=(
                        f"Research timed out after {iteration} rounds. "
                        f"Evidence gathered but synthesis failed: {synth_error}"
                    ),
                    data={"reason": "timeout_synthesis_failed", "iterations": iteration},
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

    def _smart_truncate(self, text: str, max_len: int = 200) -> str:
        """Truncate at sentence boundary to avoid cutting words."""
        if len(text) <= max_len:
            return text
        # Find last sentence boundary before limit
        truncated = text[:max_len]
        last_period = truncated.rfind(". ")
        if last_period > max_len // 2:
            return truncated[: last_period + 1]
        # Fallback to word boundary
        return truncated.rsplit(" ", 1)[0] + "..."

    def _process_event(self, event: Any, iteration: int) -> AgentEvent | None:
        """Process workflow event into AgentEvent."""
        if isinstance(event, MagenticOrchestratorMessageEvent):
            # FILTERING: Skip internal framework bookkeeping
            if event.kind in ("task_ledger", "instruction"):
                return None

            # TRANSFORMATION: Handle user_task BEFORE text extraction
            # (user_task uses static message, doesn't need text content)
            if event.kind == "user_task":
                return AgentEvent(
                    type="progress",
                    message="Manager assigning research task to agents...",
                    iteration=iteration,
                )

            # For other manager events, extract and validate text
            text = self._extract_text(event.message)
            if not text:
                return None

            # Default fallback for other manager events
            return AgentEvent(
                type="judging",
                message=f"Manager ({event.kind}): {self._smart_truncate(text)}",
                iteration=iteration,
            )

        elif isinstance(event, MagenticAgentMessageEvent):
            agent_name = event.agent_id or "unknown"
            text = self._extract_text(event.message)
            event_type = self._get_event_type_for_agent(agent_name)

            # All returned types are valid AgentEvent.type literals
            return AgentEvent(
                type=event_type,  # type: ignore[arg-type]
                message=f"{agent_name}: {self._smart_truncate(text)}",
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
