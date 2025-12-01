"""Orchestrators package - Unified Architecture (SPEC-16).

This package implements the Strategy Pattern with a unified orchestration approach:

- Advanced: Multi-agent coordination using Microsoft Agent Framework (DEFAULT)
  - Backend auto-selects: OpenAI (if key) → HuggingFace (free fallback)
- Hierarchical: Sub-iteration middleware with fine-grained control

Unified Architecture (SPEC-16):
  All users get Advanced Mode. The chat client factory auto-selects the backend:
  - With OpenAI key → OpenAIChatClient (GPT-5)
  - Without key → HuggingFaceChatClient (Llama 3.1 70B, free tier)

Usage:
    from src.orchestrators import create_orchestrator

    # Creates AdvancedOrchestrator with auto-selected backend
    orchestrator = create_orchestrator()

    # Or with explicit API key
    orchestrator = create_orchestrator(api_key="sk-...")

Protocols:
    from src.orchestrators import SearchHandlerProtocol, JudgeHandlerProtocol, OrchestratorProtocol

Design Patterns Applied:
- Factory Pattern: create_orchestrator() creates appropriate orchestrator
- Adapter Pattern: HuggingFaceChatClient adapts HF API to BaseChatClient
- Strategy Pattern: Different backends (OpenAI, HuggingFace) via ChatClientFactory
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
    "OrchestratorProtocol",
    "SearchHandlerProtocol",
    "create_orchestrator",
    "get_advanced_orchestrator",
    "get_hierarchical_orchestrator",
    "get_magentic_orchestrator",
]
