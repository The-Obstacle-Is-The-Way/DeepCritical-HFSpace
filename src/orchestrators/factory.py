"""Factory for creating orchestrators.

Implements the Factory Pattern (GoF) for creating the appropriate
orchestrator based on configuration and available credentials.

Design Principles:
- Open/Closed: Easy to add new orchestrator types without modifying existing code
- Dependency Inversion: Returns protocol-compatible objects, not concrete types
- Single Responsibility: Only handles orchestrator creation logic
"""

from typing import TYPE_CHECKING, Literal

import structlog

from src.config.domain import ResearchDomain
from src.orchestrators.base import (
    JudgeHandlerProtocol,
    OrchestratorProtocol,
    SearchHandlerProtocol,
)
from src.utils.config import settings
from src.utils.models import OrchestratorConfig

if TYPE_CHECKING:
    from src.orchestrators.advanced import AdvancedOrchestrator

logger = structlog.get_logger()


def _get_advanced_orchestrator_class() -> type["AdvancedOrchestrator"]:
    """Import AdvancedOrchestrator lazily."""
    try:
        from src.orchestrators.advanced import AdvancedOrchestrator

        return AdvancedOrchestrator
    except ImportError as e:
        logger.error("Failed to import AdvancedOrchestrator", error=str(e))
        # With unified architecture, we should never fail here unless installation is broken
        raise


def create_orchestrator(
    search_handler: SearchHandlerProtocol | None = None,
    judge_handler: JudgeHandlerProtocol | None = None,
    config: OrchestratorConfig | None = None,
    mode: Literal["simple", "magentic", "advanced", "hierarchical"] | None = None,
    api_key: str | None = None,
    domain: ResearchDomain | str | None = None,
) -> OrchestratorProtocol:
    """
    Create an orchestrator instance.

    Defaults to AdvancedOrchestrator (Unified Architecture).
    Simple Mode is deprecated and mapped to Advanced Mode.
    """
    effective_config = config or OrchestratorConfig()
    effective_mode = _determine_mode(mode)
    logger.info("Creating orchestrator", mode=effective_mode, domain=domain)

    if effective_mode == "hierarchical":
        from src.orchestrators.hierarchical import HierarchicalOrchestrator

        return HierarchicalOrchestrator(config=effective_config, domain=domain)

    # Default: Advanced Mode (Unified)
    # Handles both Paid (OpenAI) and Free (HuggingFace) tiers
    orchestrator_cls = _get_advanced_orchestrator_class()
    return orchestrator_cls(
        max_rounds=settings.advanced_max_rounds,
        api_key=api_key,
        domain=domain,
    )


def _determine_mode(explicit_mode: str | None) -> str:
    """Determine which mode to use.

    Args:
        explicit_mode: Mode explicitly requested by caller

    Returns:
        Effective mode string: "advanced" (default) or "hierarchical"
    """
    if explicit_mode == "hierarchical":
        return "hierarchical"

    # "simple" is deprecated -> upgrade to "advanced"
    # "magentic" is alias for "advanced"
    return "advanced"
