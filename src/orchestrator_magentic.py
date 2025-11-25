"""Magentic-based orchestrator for DeepCritical."""

from collections.abc import AsyncGenerator

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

from src.agents.judge_agent import JudgeAgent
from src.agents.search_agent import SearchAgent
from src.orchestrator import JudgeHandlerProtocol, SearchHandlerProtocol
from src.utils.config import settings
from src.utils.models import AgentEvent, Evidence

logger = structlog.get_logger()


class MagenticOrchestrator:
    """
    Magentic-based orchestrator - same API as Orchestrator.

    Uses Microsoft Agent Framework's MagenticBuilder for multi-agent coordination.
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

        # Initialize embedding service (optional)
        embedding_service = None
        try:
            from src.services.embeddings import get_embedding_service

            embedding_service = get_embedding_service()
            logger.info("Embedding service enabled")
        except ImportError:
            logger.info("Embedding service not available (dependencies missing)")
        except Exception as e:
            logger.warning("Failed to initialize embedding service", error=str(e))

        # Create agent wrappers
        search_agent = SearchAgent(
            self._search_handler, self._evidence_store, embedding_service=embedding_service
        )
        judge_agent = JudgeAgent(self._judge_handler, self._evidence_store)

        # Build Magentic workflow
        # Note: MagenticBuilder.participants takes named arguments for agent instances
        workflow = (
            MagenticBuilder()
            .participants(
                searcher=search_agent,
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

        # Task instruction for the manager
        semantic_note = ""
        if embedding_service:
            semantic_note = """
The system has semantic search enabled. When evidence is found:
1. Related concepts will be automatically surfaced
2. Duplicates are removed by meaning, not just URL
3. Use the surfaced related concepts to refine searches
"""

        task = f"""Research drug repurposing opportunities for: {query}
{semantic_note}
Instructions:
1. Use SearcherAgent to find evidence. SEND ONLY A SIMPLE KEYWORD QUERY (e.g. "metformin aging")
   as the instruction. Complex queries fail.
2. Use JudgeAgent to evaluate if evidence is sufficient.
3. If JudgeAgent says "continue", search with refined queries.
4. If JudgeAgent says "synthesize", provide final synthesis
5. Stop when synthesis is ready or max rounds reached

Focus on finding:
- Mechanism of action evidence
- Clinical/preclinical studies
- Specific drug candidates
"""

        iteration = 0
        try:
            # workflow.run_stream returns an async generator of workflow events
            # We use 'await' in the for loop for async generator
            async for event in workflow.run_stream(task):
                if isinstance(event, MagenticOrchestratorMessageEvent):
                    # Manager events (planning, instruction, ledger)
                    # The 'message' attribute might be None if it's just a state change,
                    # check message presence
                    message_text = (
                        event.message.text
                        if event.message and hasattr(event.message, "text")
                        else ""
                    )
                    # kind might be 'plan', 'instruction', etc.
                    kind = getattr(event, "kind", "manager")

                    if message_text:
                        yield AgentEvent(
                            type="judging",
                            message=f"Manager ({kind}): {message_text[:100]}...",
                            iteration=iteration,
                        )

                elif isinstance(event, MagenticAgentMessageEvent):
                    # Complete agent response
                    iteration += 1
                    agent_name = event.agent_id or "unknown"
                    msg_text = (
                        event.message.text
                        if event.message and hasattr(event.message, "text")
                        else ""
                    )

                    if "search" in agent_name.lower():
                        # Check if we found evidence (based on SearchAgent logic)
                        yield AgentEvent(
                            type="search_complete",
                            message=f"Search agent: {msg_text[:100]}...",
                            iteration=iteration,
                        )
                    elif "judge" in agent_name.lower():
                        yield AgentEvent(
                            type="judge_complete",
                            message=f"Judge agent: {msg_text[:100]}...",
                            iteration=iteration,
                        )

                elif isinstance(event, MagenticFinalResultEvent):
                    # Final workflow result
                    final_text = (
                        event.message.text
                        if event.message and hasattr(event.message, "text")
                        else "No result"
                    )
                    yield AgentEvent(
                        type="complete",
                        message=final_text,
                        data={"iterations": iteration},
                        iteration=iteration,
                    )

                elif isinstance(event, MagenticAgentDeltaEvent):
                    # Streaming token chunks from agents (optional "typing" effect)
                    # Only emit if we have actual text content
                    if event.text:
                        yield AgentEvent(
                            type="streaming",
                            message=event.text,
                            data={"agent_id": event.agent_id},
                            iteration=iteration,
                        )

                elif isinstance(event, WorkflowOutputEvent):
                    # Alternative final output event
                    if event.data:
                        yield AgentEvent(
                            type="complete",
                            message=str(event.data),
                            iteration=iteration,
                        )

        except Exception as e:
            logger.error("Magentic workflow failed", error=str(e))
            yield AgentEvent(
                type="error",
                message=f"Workflow error: {e!s}",
                iteration=iteration,
            )
