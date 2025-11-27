"""Factory for creating orchestrators."""

from typing import Any, Literal

from src.orchestrator import JudgeHandlerProtocol, Orchestrator, SearchHandlerProtocol
from src.utils.models import OrchestratorConfig


def create_orchestrator(
    search_handler: SearchHandlerProtocol | None = None,
    judge_handler: JudgeHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic"] = "simple",
) -> Any:
    """
    Create an orchestrator instance.

    Args:
        search_handler: The search handler (required for simple mode)
        judge_handler: The judge handler (required for simple mode)
        config: Optional configuration
        mode: "simple" for Phase 4 loop, "magentic" for ChatAgent-based multi-agent

    Returns:
        Orchestrator instance

    Note:
        Magentic mode does NOT use search_handler/judge_handler.
        It creates ChatAgent instances with internal LLMs that call tools directly.
    """
    if mode == "magentic":
        try:
            from src.orchestrator_magentic import MagenticOrchestrator

            return MagenticOrchestrator(
                max_rounds=config.max_iterations if config else 10,
            )
        except ImportError:
            # Fallback to simple if agent-framework not installed
            pass

    # Simple mode requires handlers
    if search_handler is None or judge_handler is None:
        raise ValueError("Simple mode requires search_handler and judge_handler")

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )
