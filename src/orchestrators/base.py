"""Base protocols and shared types for orchestrators.

This module defines the interfaces that orchestrators depend on,
following the Interface Segregation Principle (ISP) and
Dependency Inversion Principle (DIP).
"""

from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from src.utils.models import AgentEvent, Evidence, JudgeAssessment, SearchResult


class SearchHandlerProtocol(Protocol):
    """Protocol for search handler.

    Defines the interface for executing searches across biomedical databases.
    Implementations include SearchHandler (scatter-gather across PubMed,
    ClinicalTrials.gov, Europe PMC).
    """

    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        """Execute a search query.

        Args:
            query: The search query string
            max_results_per_tool: Maximum results to fetch per search tool

        Returns:
            SearchResult containing evidence and metadata
        """
        ...


class JudgeHandlerProtocol(Protocol):
    """Protocol for judge handler.

    Defines the interface for assessing evidence quality and sufficiency.
    Implementations include JudgeHandler (pydantic-ai), HFInferenceJudgeHandler,
    and MockJudgeHandler.
    """

    async def assess(
        self,
        question: str,
        evidence: list[Evidence],
        iteration: int = 0,
        max_iterations: int = 10,
    ) -> JudgeAssessment:
        """Assess whether collected evidence is sufficient.

        Args:
            question: The original research question
            evidence: List of evidence items to assess
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            JudgeAssessment with sufficiency determination and next steps
        """
        ...


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Protocol for orchestrators.

    All orchestrators (Simple, Advanced, Hierarchical) implement this interface,
    allowing them to be used interchangeably by the factory and UI.
    """

    def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the orchestrator workflow.

        Args:
            query: User's research question

        Yields:
            AgentEvent objects for real-time UI updates
        """
        ...
