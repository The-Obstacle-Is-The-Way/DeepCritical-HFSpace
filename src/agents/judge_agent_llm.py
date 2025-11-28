"""LLM Judge for sub-iterations."""

from typing import Any

import structlog
from pydantic_ai import Agent

from src.agent_factory.judges import get_model
from src.utils.models import JudgeAssessment

logger = structlog.get_logger()


class LLMSubIterationJudge:
    """Judge that uses an LLM to assess sub-iteration results."""

    def __init__(self) -> None:
        self.model = get_model()
        self.agent = Agent(
            model=self.model,
            output_type=JudgeAssessment,
            system_prompt="""You are a strict judge evaluating a research task.

Evaluate if the result is sufficient to answer the task.
Provide scores and detailed reasoning.
If not sufficient, suggest next steps.""",
            retries=3,
        )

    async def assess(self, task: str, result: Any, history: list[Any]) -> JudgeAssessment:
        """Assess the result using LLM."""
        logger.info("LLM judge assessing result", task=task[:100], history_len=len(history))

        prompt = f"""Task: {task}

Current Result:
{str(result)[:4000]}

History of previous attempts: {len(history)}

Evaluate validity and sufficiency."""

        run_result = await self.agent.run(prompt)
        logger.info("LLM judge assessment complete", sufficient=run_result.output.sufficient)
        return run_result.output
