"""Hierarchical orchestrator using middleware and sub-teams."""

import asyncio
from collections.abc import AsyncGenerator

import structlog

from src.agents.judge_agent_llm import LLMSubIterationJudge
from src.agents.magentic_agents import create_search_agent
from src.middleware.sub_iteration import SubIterationMiddleware, SubIterationTeam
from src.services.embeddings import get_embedding_service
from src.state import init_magentic_state
from src.utils.models import AgentEvent

logger = structlog.get_logger()


class ResearchTeam(SubIterationTeam):
    """Adapts Magentic ChatAgent to SubIterationTeam protocol."""

    def __init__(self) -> None:
        self.agent = create_search_agent()

    async def execute(self, task: str) -> str:
        response = await self.agent.run(task)
        if response.messages:
            for msg in reversed(response.messages):
                if msg.role == "assistant" and msg.text:
                    return str(msg.text)
        return "No response from agent."


class HierarchicalOrchestrator:
    """Orchestrator that uses hierarchical teams and sub-iterations."""

    def __init__(self) -> None:
        self.team = ResearchTeam()
        self.judge = LLMSubIterationJudge()
        self.middleware = SubIterationMiddleware(self.team, self.judge, max_iterations=5)

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        logger.info("Starting hierarchical orchestrator", query=query)

        try:
            service = get_embedding_service()
            init_magentic_state(service)
        except Exception as e:
            logger.warning(
                "Embedding service initialization failed, using default state",
                error=str(e),
            )
            init_magentic_state()

        yield AgentEvent(type="started", message=f"Starting research: {query}")

        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def event_callback(event: AgentEvent) -> None:
            await queue.put(event)

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

        try:
            result, assessment = await task_future

            assessment_text = assessment.reasoning if assessment else "None"
            yield AgentEvent(
                type="complete",
                message=(
                    f"Research complete.\n\nResult:\n{result}\n\nAssessment:\n{assessment_text}"
                ),
                data={"assessment": assessment.model_dump() if assessment else None},
            )
        except Exception as e:
            logger.error("Orchestrator failed", error=str(e))
            yield AgentEvent(type="error", message=f"Orchestrator failed: {e}")
