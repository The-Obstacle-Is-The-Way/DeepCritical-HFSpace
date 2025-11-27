"""Middleware for orchestrating sub-iterations with research teams and judges."""

from typing import Any, Protocol

import structlog

from src.utils.models import AgentEvent, JudgeAssessment

logger = structlog.get_logger()


class SubIterationTeam(Protocol):
    """Protocol for a research team that executes a sub-task."""

    async def execute(self, task: str) -> Any:
        """Execute the sub-task and return a result."""
        ...


class SubIterationJudge(Protocol):
    """Protocol for a judge that evaluates the sub-task result."""

    async def assess(self, task: str, result: Any, history: list[Any]) -> JudgeAssessment:
        """Assess the quality of the result."""
        ...


class SubIterationMiddleware:
    """
    Middleware that manages a sub-iteration loop:
    1. Orchestrator delegates to a Research Team.
    2. Research Team produces a result.
    3. Judge evaluates the result.
    4. Loop continues until Judge approves or max iterations reached.
    """

    def __init__(
        self,
        team: SubIterationTeam,
        judge: SubIterationJudge,
        max_iterations: int = 3,
    ):
        self.team = team
        self.judge = judge
        self.max_iterations = max_iterations

    async def run(
        self,
        task: str,
        event_callback: Any = None,  # Optional callback for streaming events
    ) -> tuple[Any, JudgeAssessment | None]:
        """
        Run the sub-iteration loop.

        Args:
            task: The research task or question.
            event_callback: Async callable to report events (e.g. to UI).

        Returns:
            Tuple of (best_result, final_assessment).
        """
        history: list[Any] = []
        best_result: Any = None
        final_assessment: JudgeAssessment | None = None

        for i in range(1, self.max_iterations + 1):
            logger.info("Sub-iteration starting", iteration=i, task=task)

            if event_callback:
                await event_callback(
                    AgentEvent(
                        type="looping",
                        message=f"Sub-iteration {i}: Executing task...",
                        iteration=i,
                    )
                )

            # 1. Team Execution
            try:
                result = await self.team.execute(task)
                history.append(result)
                best_result = result  # Assume latest is best for now
            except Exception as e:
                logger.error("Sub-iteration execution failed", error=str(e))
                if event_callback:
                    await event_callback(
                        AgentEvent(
                            type="error",
                            message=f"Sub-iteration execution failed: {e}",
                            iteration=i,
                        )
                    )
                return best_result, final_assessment

            # 2. Judge Assessment
            try:
                assessment = await self.judge.assess(task, result, history)
                final_assessment = assessment
            except Exception as e:
                logger.error("Sub-iteration judge failed", error=str(e))
                if event_callback:
                    await event_callback(
                        AgentEvent(
                            type="error",
                            message=f"Sub-iteration judge failed: {e}",
                            iteration=i,
                        )
                    )
                return best_result, final_assessment

            # 3. Decision
            if assessment.sufficient:
                logger.info("Sub-iteration sufficient", iteration=i)
                return best_result, assessment

            # If not sufficient, we might refine the task for the next iteration
            # For this implementation, we assume the team is smart enough or the task stays same
            # but we could append feedback to the task.

            feedback = assessment.reasoning
            logger.info("Sub-iteration insufficient", feedback=feedback)

            if event_callback:
                await event_callback(
                    AgentEvent(
                        type="looping",
                        message=(
                            f"Sub-iteration {i} result insufficient. "
                            f"Feedback: {feedback[:100]}..."
                        ),
                        iteration=i,
                    )
                )

        logger.warning("Sub-iteration max iterations reached", task=task)
        return best_result, final_assessment
