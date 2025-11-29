"""Orchestrators package - provides different orchestration strategies.

This package implements the Strategy Pattern, allowing the application
to switch between different orchestration approaches:

- Simple: Basic search-judge loop using pydantic-ai (free tier compatible)
- Advanced: Multi-agent coordination using Microsoft Agent Framework
- Hierarchical: Sub-iteration middleware with fine-grained control

Usage:
    from src.orchestrators import create_orchestrator, Orchestrator

    # Auto-detect mode based on available API keys
    orchestrator = create_orchestrator(search_handler, judge_handler)

    # Or explicitly specify mode
    orchestrator = create_orchestrator(mode="advanced", api_key="sk-...")

Protocols:
    from src.orchestrators import SearchHandlerProtocol, JudgeHandlerProtocol, OrchestratorProtocol

Design Patterns Applied:
- Factory Pattern: create_orchestrator() creates appropriate orchestrator
- Strategy Pattern: Different orchestrators implement different strategies
- Facade Pattern: This __init__.py provides a clean public API
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Protocols (Interface Segregation Principle)
from src.orchestrators.base import (
    JudgeHandlerProtocol,
    OrchestratorProtocol,
    SearchHandlerProtocol,
)

# Factory (creational pattern)
from src.orchestrators.factory import create_orchestrator

# Orchestrators (Strategy Pattern implementations)
from src.orchestrators.simple import Orchestrator

if TYPE_CHECKING:
    from src.orchestrators.advanced import AdvancedOrchestrator
    from src.orchestrators.hierarchical import HierarchicalOrchestrator

# Lazy imports for optional dependencies
# These are not imported at module level to avoid breaking simple mode
# when agent-framework-core is not installed


def get_advanced_orchestrator() -> type[AdvancedOrchestrator]:
    """Get the AdvancedOrchestrator class (requires agent-framework-core).

    Returns:
        The AdvancedOrchestrator class

    Raises:
        ImportError: If agent-framework-core is not installed
    """
    from src.orchestrators.advanced import AdvancedOrchestrator

    return AdvancedOrchestrator


def get_hierarchical_orchestrator() -> type[HierarchicalOrchestrator]:
    """Get the HierarchicalOrchestrator class (requires agent-framework-core).

    Returns:
        The HierarchicalOrchestrator class

    Raises:
        ImportError: If agent-framework-core is not installed
    """
    from src.orchestrators.hierarchical import HierarchicalOrchestrator

    return HierarchicalOrchestrator


def get_magentic_orchestrator() -> type[AdvancedOrchestrator]:
    """Get the AdvancedOrchestrator class.

    .. deprecated:: 0.1.0
        Use :func:`get_advanced_orchestrator` instead.
        The name 'magentic' was confusing with the 'magentic' PyPI package.

    Returns:
        The AdvancedOrchestrator class
    """
    warnings.warn(
        "get_magentic_orchestrator() is deprecated, use get_advanced_orchestrator() instead. "
        "The name 'magentic' was confusing with the 'magentic' PyPI package.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_advanced_orchestrator()


__all__ = [
    "JudgeHandlerProtocol",
    "Orchestrator",
    "OrchestratorProtocol",
    "SearchHandlerProtocol",
    "create_orchestrator",
    "get_advanced_orchestrator",
    "get_hierarchical_orchestrator",
    "get_magentic_orchestrator",
]
