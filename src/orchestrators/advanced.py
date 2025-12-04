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
from typing import TYPE_CHECKING, Any, Literal

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
from src.agents.state import get_magentic_state, init_magentic_state
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

# Agent ID constants - prevents silent breakage if agents are renamed
REPORTER_AGENT_ID = "reporter"
SEARCHER_AGENT_ID = "searcher"
JUDGE_AGENT_ID = "judge"
HYPOTHESIZER_AGENT_ID = "hypothesizer"


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

        # Store API key for service initialization
        self._api_key = api_key

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
        return get_embedding_service_if_available(api_key=self._api_key)

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

    async def _init_workflow_events(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Yield initialization events."""
        yield AgentEvent(
            type="started",
            message=f"Starting research (Advanced mode): {query}",
            iteration=0,
        )

        yield AgentEvent(
            type="progress",
            message="Loading embedding service (LlamaIndex/ChromaDB)...",
            iteration=0,
        )

    async def _synthesize_fallback(
        self, iteration: int, reason: str
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Unified fallback synthesis for all termination scenarios.

        This method handles synthesis when the workflow terminates without
        a proper report from ReportAgent. It's a safety net for:
        - Timeout scenarios
        - Manager model failing to delegate to ReportAgent (7B model limitation)
        - Max rounds reached without synthesis

        Args:
            iteration: Current workflow iteration count
            reason: Why synthesis is being forced ("timeout", "no_reporter", "max_rounds")
        """
        status_messages = {
            "timeout": "Workflow timed out. Synthesizing available evidence...",
            "no_reporter": "Synthesizing research findings...",
            "max_rounds": "Max rounds reached. Synthesizing findings...",
        }

        try:
            state = get_magentic_state()
            evidence_summary = await state.memory.get_context_summary()
            report_agent = create_report_agent(self._chat_client, domain=self.domain)

            yield AgentEvent(
                type="synthesizing",
                message=status_messages.get(reason, "Synthesizing..."),
                iteration=iteration,
            )

            synthesis_result = await report_agent.run(
                "Synthesize research report from this evidence. "
                f"If evidence is sparse, say so.\n\n{evidence_summary}"
            )

            yield AgentEvent(
                type="complete",
                message=synthesis_result.text,
                data={"reason": f"{reason}_synthesis", "iterations": iteration},
                iteration=iteration,
            )
        except Exception as synth_error:
            logger.error(f"{reason} synthesis failed", error=str(synth_error))
            yield AgentEvent(
                type="complete",
                message=f"Research completed. Synthesis failed: {synth_error}",
                data={"reason": f"{reason}_synthesis_failed", "iterations": iteration},
                iteration=iteration,
            )

    async def run(  # noqa: PLR0915 - Complex but necessary for event stream handling
        self, query: str
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        logger.info("Starting Advanced orchestrator", query=query)

        async for event in self._init_workflow_events(query):
            yield event

        # Initialize context state
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
        reporter_ran = False  # P1 FIX: Track if ReportAgent produced output

        # ACCUMULATOR PATTERN: Track streaming content to bypass upstream Repr Bug
        # Upstream bug in _magentic.py flattens message.contents and sets message.text
        # to repr(message) if text is empty. We must reconstruct text from Deltas.
        current_message_buffer: str = ""
        current_agent_id: str | None = None
        last_streamed_length: int = 0  # Track for P2 Duplicate Report Bug Fix

        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for event in workflow.run_stream(task):
                    # 1. Handle Streaming (Source of Truth for Content)
                    if isinstance(event, MagenticAgentDeltaEvent):
                        # Detect agent switch to clear buffer
                        if event.agent_id != current_agent_id:
                            current_message_buffer = ""
                            current_agent_id = event.agent_id

                        if event.text:
                            current_message_buffer += event.text
                            yield AgentEvent(
                                type="streaming",
                                message=event.text,
                                data={"agent_id": event.agent_id},
                                iteration=iteration,
                            )
                        continue

                    # 2. Handle Completion Signal
                    # We use our accumulated buffer instead of the corrupted event.message
                    if isinstance(event, MagenticAgentMessageEvent):
                        iteration += 1

                        # P1 FIX: Track if ReportAgent produced output
                        agent_name = (event.agent_id or "").lower()
                        if REPORTER_AGENT_ID in agent_name:
                            reporter_ran = True

                        comp_event, prog_event = self._handle_completion_event(
                            event, current_message_buffer, iteration
                        )
                        yield comp_event
                        yield prog_event

                        # P2 BUG FIX: Save length before clearing
                        last_streamed_length = len(current_message_buffer)
                        # Clear buffer after consuming
                        current_message_buffer = ""
                        continue

                    # 3. Handle Final Events Inline (P2 Duplicate Report Fix + P1 Forced Synthesis)
                    if isinstance(event, (MagenticFinalResultEvent, WorkflowOutputEvent)):
                        if final_event_received:
                            continue  # Skip duplicate final events
                        final_event_received = True

                        # P1 FIX: Force synthesis if ReportAgent never ran
                        if not reporter_ran:
                            logger.warning(
                                "ReportAgent never ran - forcing synthesis",
                                iterations=iteration,
                            )
                            async for synth_event in self._synthesize_fallback(
                                iteration, "no_reporter"
                            ):
                                yield synth_event
                        else:
                            yield self._handle_final_event(event, iteration, last_streamed_length)
                        continue

                    # 4. Handle other events normally
                    agent_event = self._process_event(event, iteration)
                    if agent_event:
                        yield agent_event

            # GUARANTEE: Always emit termination event if stream ends without one
            # (e.g., max rounds reached)
            if not final_event_received:
                logger.warning(
                    "Workflow ended without final event",
                    iterations=iteration,
                )
                # P1 FIX: Force synthesis if ReportAgent never ran
                if not reporter_ran:
                    async for synth_event in self._synthesize_fallback(iteration, "max_rounds"):
                        yield synth_event
                else:
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
            async for event in self._synthesize_fallback(iteration, "timeout"):
                yield event

        except Exception as e:
            logger.error("Workflow failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Workflow error: {e!s}",
                iteration=iteration,
            )

    def _handle_completion_event(
        self, event: MagenticAgentMessageEvent, buffer: str, iteration: int
    ) -> tuple[AgentEvent, AgentEvent]:
        """Handle an agent completion event using the accumulated buffer."""
        # Use buffer if available, otherwise fall back cautiously
        # (Only fall back if buffer empty, which implies tool-only turn)
        text_content = buffer
        if not text_content:
            # Try extraction but ignore repr strings AND empty strings
            raw_text = self._extract_text(event.message)
            if raw_text and not (raw_text.startswith("<") and "object at" in raw_text):
                text_content = raw_text
            else:
                text_content = "Action completed (Tool Call)"

        agent_name = event.agent_id or "unknown"
        event_type = self._get_event_type_for_agent(agent_name)

        completion_event = AgentEvent(
            type=event_type,
            message=f"{agent_name}: {text_content[:200]}...",
            iteration=iteration,
        )

        # Progress update
        rounds_remaining = max(self._max_rounds - iteration, 0)
        est_seconds = rounds_remaining * 45
        est_display = (
            f"{est_seconds // 60}m {est_seconds % 60}s" if est_seconds >= 60 else f"{est_seconds}s"
        )

        progress_event = AgentEvent(
            type="progress",
            message=f"Round {iteration}/{self._max_rounds} (~{est_display} remaining)",
            iteration=iteration,
        )

        return completion_event, progress_event

    def _handle_final_event(
        self,
        event: MagenticFinalResultEvent | WorkflowOutputEvent,
        iteration: int,
        last_streamed_length: int,
    ) -> AgentEvent:
        """Handle final workflow events with duplicate content suppression (P2 Bug Fix)."""
        # DECISION: Did we stream substantial content?
        if last_streamed_length > 100:
            # YES: Final event is a SIGNAL, not a payload
            return AgentEvent(
                type="complete",
                message="Research complete.",
                data={
                    "iterations": iteration,
                    "streamed_chars": last_streamed_length,
                },
                iteration=iteration,
            )

        # NO: Final event must carry the payload (tool-only turn, cache hit)
        text = ""
        if isinstance(event, MagenticFinalResultEvent):
            text = self._extract_text(event.message) if event.message else "No result"
        elif isinstance(event, WorkflowOutputEvent):
            text = self._extract_text(event.data) if event.data else "Research complete"

        return AgentEvent(
            type="complete",
            message=text,
            data={"iterations": iteration},
            iteration=iteration,
        )

    def _extract_text(self, message: Any) -> str:
        """
        Defensively extract text from a message object.

        Handles ChatMessage objects from both OpenAI and HuggingFace clients.
        ChatMessage has: .text (str), .contents (list of content objects)
        Also handles plain string messages (e.g., WorkflowOutputEvent.data).
        """
        if not message:
            return ""

        # Priority 0: Handle plain string messages (e.g., WorkflowOutputEvent.data)
        if isinstance(message, str):
            # Filter out obvious repr-style noise
            if not (message.startswith("<") and "object at" in message):
                return message
            return ""

        # Priority 1: .text (standard ChatMessage text content)
        if hasattr(message, "text") and message.text:
            text = message.text
            # Verify it's actually a string, not the object itself
            if isinstance(text, str) and not (text.startswith("<") and "object at" in text):
                return text

        # Priority 2: .contents (list of FunctionCallContent, TextContent, etc.)
        # This handles tool call responses from HuggingFace
        if hasattr(message, "contents") and message.contents:
            parts = []
            for content in message.contents:
                # TextContent has .text
                if hasattr(content, "text") and content.text:
                    parts.append(str(content.text))
                # FunctionCallContent has .name and .arguments
                elif hasattr(content, "name"):
                    parts.append(f"[Tool: {content.name}]")
            if parts:
                return " ".join(parts)

        # Priority 3: .content (legacy - some frameworks use singular)
        if hasattr(message, "content") and message.content:
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join([str(c.text) for c in content if hasattr(c, "text")])

        # Fallback: Return empty string instead of repr
        # The repr is useless for display purposes
        return ""

    def _get_event_type_for_agent(
        self, agent_name: str
    ) -> Literal["search_complete", "judge_complete", "hypothesizing", "synthesizing", "judging"]:
        """Map agent name to appropriate event type.

        Args:
            agent_name: The agent ID from the workflow event

        Returns:
            Event type string matching AgentEvent.type Literal
        """
        agent_lower = agent_name.lower()
        if SEARCHER_AGENT_ID in agent_lower:
            return "search_complete"
        if JUDGE_AGENT_ID in agent_lower:
            return "judge_complete"
        if HYPOTHESIZER_AGENT_ID in agent_lower:
            return "hypothesizing"
        if REPORTER_AGENT_ID in agent_lower:
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

        # NOTE: The following event types are handled inline in run() loop and never reach
        # this method due to `continue` statements:
        # - MagenticAgentMessageEvent: Accumulator Pattern (lines 322-335)
        # - MagenticAgentDeltaEvent: Accumulator Pattern (lines 306-320)
        # - MagenticFinalResultEvent: P2 Duplicate Fix via _handle_final_event() (lines 343-347)
        # - WorkflowOutputEvent: P2 Duplicate Fix via _handle_final_event() (lines 343-347)

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
