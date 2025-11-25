"""Factory for creating orchestrators."""

from typing import Any, Literal

from src.orchestrator import JudgeHandlerProtocol, Orchestrator, SearchHandlerProtocol
from src.utils.models import OrchestratorConfig

# Define protocols again or import if they were in a shared place.

# Since they are in src/orchestrator.py, we can import them?

# But SearchHandler and JudgeHandler in arguments are concrete classes in the type hint,

# which satisfy the protocol.


def create_orchestrator(
    search_handler: SearchHandlerProtocol,
    judge_handler: JudgeHandlerProtocol,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic"] = "simple",
) -> Any:
    """
    Create an orchestrator instance.

    Args:
        search_handler: The search handler
        judge_handler: The judge handler
        config: Optional configuration
        mode: "simple" for Phase 4 loop, "magentic" for Phase 5 multi-agent

    Returns:
        Orchestrator instance (same interface regardless of mode)
    """
    if mode == "magentic":
        try:
            from src.orchestrator_magentic import MagenticOrchestrator

            return MagenticOrchestrator(
                search_handler=search_handler,
                judge_handler=judge_handler,
                max_rounds=config.max_iterations if config else 10,
            )
        except ImportError:
            # Fallback to simple if agent-framework not installed
            pass

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )
