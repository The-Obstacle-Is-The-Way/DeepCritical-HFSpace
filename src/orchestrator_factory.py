"""Factory for creating orchestrators."""

from typing import Any, Literal

import structlog

from src.orchestrator import JudgeHandlerProtocol, Orchestrator, SearchHandlerProtocol
from src.utils.config import settings
from src.utils.models import OrchestratorConfig

logger = structlog.get_logger()


def _get_magentic_orchestrator_class() -> Any:
    """Import MagenticOrchestrator lazily to avoid hard dependency."""
    try:
        from src.orchestrator_magentic import MagenticOrchestrator

        return MagenticOrchestrator
    except ImportError as e:
        logger.error("Failed to import MagenticOrchestrator", error=str(e))
        raise ValueError(
            "Advanced mode requires agent-framework-core. Please install it or use mode='simple'."
        ) from e


def create_orchestrator(
    search_handler: SearchHandlerProtocol | None = None,
    judge_handler: JudgeHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic", "advanced"] | None = None,
    api_key: str | None = None,
) -> Any:
    """
    Create an orchestrator instance.

    Args:
        search_handler: The search handler (required for simple mode)
        judge_handler: The judge handler (required for simple mode)
        config: Optional configuration
        mode: "simple", "magentic", "advanced" or None (auto-detect)
        api_key: Optional API key for advanced mode (OpenAI)

    Returns:
        Orchestrator instance
    """
    effective_mode = _determine_mode(mode, api_key)
    logger.info("Creating orchestrator", mode=effective_mode)

    if effective_mode == "advanced":
        orchestrator_cls = _get_magentic_orchestrator_class()
        return orchestrator_cls(
            max_rounds=config.max_iterations if config else 10,
            api_key=api_key,
        )

    # Simple mode requires handlers
    if search_handler is None or judge_handler is None:
        raise ValueError("Simple mode requires search_handler and judge_handler")

    return Orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
    )


def _determine_mode(explicit_mode: str | None, api_key: str | None) -> str:
    """Determine which mode to use."""
    if explicit_mode:
        if explicit_mode in ("magentic", "advanced"):
            return "advanced"
        return "simple"

    # Auto-detect: advanced if paid API key available
    if settings.has_openai_key or (api_key and api_key.startswith("sk-")):
        return "advanced"

    return "simple"
