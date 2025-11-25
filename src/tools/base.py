"""Base classes and protocols for search tools."""

from typing import Protocol

from src.utils.models import Evidence


class SearchTool(Protocol):
    """Protocol defining the interface for all search tools."""

    @property
    def name(self) -> str:
        """Human-readable name of this tool."""
        ...

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Execute a search and return evidence.

        Args:
            query: The search query string
            max_results: Maximum number of results to return

        Returns:
            List of Evidence objects

        Raises:
            SearchError: If the search fails
            RateLimitError: If we hit rate limits
        """
        ...
