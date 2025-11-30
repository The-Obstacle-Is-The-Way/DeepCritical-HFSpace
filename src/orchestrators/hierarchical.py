"""Hierarchical Orchestrator using middleware and sub-teams.

This orchestrator implements a hierarchical team pattern where sub-teams
can be composed and coordinated through middleware. It provides more
granular control over the research workflow compared to the simple
orchestrator.

Design Patterns:
- Composite: Teams can contain sub-teams
- Chain of Responsibility: Middleware processes requests in sequence
- Template Method: SubIterationMiddleware defines the iteration skeleton
"""

import asyncio
from collections.abc import AsyncGenerator

import structlog

from src.agents.judge_agent_llm import LLMSubIterationJudge
from src.agents.magentic_agents import create_search_agent
from src.config.domain import ResearchDomain
from src.middleware.sub_iteration import SubIterationMiddleware, SubIterationTeam
from src.orchestrators.base import OrchestratorProtocol
from src.state import init_magentic_state
from src.utils.models import AgentEvent, OrchestratorConfig
from src.utils.service_loader import get_embedding_service_if_available

logger = structlog.get_logger()

# Default timeout for hierarchical orchestrator (5 minutes)
DEFAULT_TIMEOUT_SECONDS = 300.0


class ResearchTeam(SubIterationTeam):
    """Adapts ChatAgent to SubIterationTeam protocol.

    This adapter allows the search agent to be used within the
    sub-iteration middleware framework.
    """

    def __init__(self, domain: ResearchDomain | str | None = None) -> None:
        self.agent = create_search_agent(domain=domain)

    async def execute(self, task: str) -> str:
        """Execute a research task.

        Args:
            task: The research task description

        Returns:
            Text response from the agent
        """
        response = await self.agent.run(task)
        if response.messages:
            for msg in reversed(response.messages):
                if msg.role == "assistant" and msg.text:
                    return str(msg.text)
        return "No response from agent."


class HierarchicalOrchestrator(OrchestratorProtocol):
    """Orchestrator that uses hierarchical teams and sub-iterations.

    This orchestrator provides:
    - Sub-iteration middleware for fine-grained control
    - LLM-based judge for sub-iteration decisions
    - Event-driven architecture for UI updates
    - Configurable iterations and timeout
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        domain: ResearchDomain | str | None = None,
    ) -> None:
        """Initialize the hierarchical orchestrator.

        Args:
            config: Optional configuration (uses defaults if not provided)
            timeout_seconds: Maximum workflow duration (default: 5 minutes)
            domain: Research domain for customization
        """
        self.config = config or OrchestratorConfig()
        self._timeout_seconds = timeout_seconds
        self.domain = domain
        self.team = ResearchTeam(domain=domain)
        self.judge = LLMSubIterationJudge()
        self.middleware = SubIterationMiddleware(
            self.team, self.judge, max_iterations=self.config.max_iterations
        )

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the hierarchical workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        logger.info("Starting hierarchical orchestrator", query=query)

        service = get_embedding_service_if_available()
        init_magentic_state(query, service)

        yield AgentEvent(type="started", message=f"Starting research: {query}")

        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def event_callback(event: AgentEvent) -> None:
            await queue.put(event)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                task_future = asyncio.create_task(self.middleware.run(query, event_callback))

                while not task_future.done():
                    get_event = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait(
                        {task_future, get_event}, return_when=asyncio.FIRST_COMPLETED
                    )

                    if get_event in done:
                        event = get_event.result()
                        if event:
                            yield event
                    else:
                        get_event.cancel()

                # Process remaining events
                while not queue.empty():
                    ev = queue.get_nowait()
                    if ev:
                        yield ev

                result, assessment = await task_future

                assessment_text = assessment.reasoning if assessment else "None"
                yield AgentEvent(
                    type="complete",
                    message=(
                        f"Research complete.\n\nResult:\n{result}\n\nAssessment:\n{assessment_text}"
                    ),
                    data={"assessment": assessment.model_dump() if assessment else None},
                )

        except TimeoutError:
            logger.warning("Hierarchical workflow timed out", query=query)
            yield AgentEvent(
                type="complete",
                message="Research timed out. Results may be incomplete.",
                data={"reason": "timeout"},
            )

        except Exception as e:
            logger.error(
                "Orchestrator failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            yield AgentEvent(type="error", message=f"Orchestrator failed: {e}")
