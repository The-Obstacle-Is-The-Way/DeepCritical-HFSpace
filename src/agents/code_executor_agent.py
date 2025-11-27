"""Code execution agent using Modal."""

import asyncio

from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient

from src.tools.code_execution import get_code_executor
from src.utils.config import settings


@ai_function
async def execute_python_code(code: str) -> str:
    """Execute Python code in a secure sandbox.

    Args:
        code: The Python code to execute.

    Returns:
        The standard output and standard error of the execution.
    """
    executor = get_code_executor()
    loop = asyncio.get_running_loop()

    # Run in executor to avoid blocking
    try:
        result = await loop.run_in_executor(None, lambda: executor.execute(code))
        if result["success"]:
            return f"Stdout:\n{result['stdout']}"
        else:
            return f"Error:\n{result['error']}\nStderr:\n{result['stderr']}"
    except Exception as e:
        return f"Execution failed: {e}"


def create_code_executor_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a code executor agent.

    Args:
        chat_client: Optional custom chat client.

    Returns:
        ChatAgent configured for code execution.
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="CodeExecutorAgent",
        description="Executes Python code for data analysis, calculation, and simulation.",
        instructions="""You are a code execution expert.
When asked to analyze data or perform calculations, write Python code and execute it.
Use libraries like pandas, numpy, scipy, matplotlib.

Always output the code you want to execute using the `execute_python_code` tool.
Check the output and interpret the results.""",
        chat_client=client,
        tools=[execute_python_code],
        temperature=0.0,  # Strict code generation
    )
