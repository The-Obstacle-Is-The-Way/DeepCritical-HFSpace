"""Search handler - orchestrates multiple search tools."""

import asyncio
from typing import Literal, cast

import structlog

from src.tools.base import SearchTool
from src.utils.exceptions import SearchError
from src.utils.models import Evidence, SearchResult

logger = structlog.get_logger()


class SearchHandler:
    """Orchestrates parallel searches across multiple tools."""

    def __init__(self, tools: list[SearchTool], timeout: float = 30.0) -> None:
        """
        Initialize the search handler.

        Args:
            tools: List of search tools to use
            timeout: Timeout for each search in seconds
        """
        self.tools = tools
        self.timeout = timeout

    async def execute(self, query: str, max_results_per_tool: int = 10) -> SearchResult:
        """
        Execute search across all tools in parallel.

        Args:
            query: The search query
            max_results_per_tool: Max results from each tool

        Returns:
            SearchResult containing all evidence and metadata
        """
        logger.info("Starting search", query=query, tools=[t.name for t in self.tools])

        # Create tasks for parallel execution
        tasks = [
            self._search_with_timeout(tool, query, max_results_per_tool) for tool in self.tools
        ]

        # Gather results (don't fail if one tool fails)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_evidence: list[Evidence] = []
        sources_searched: list[Literal["pubmed"]] = []
        errors: list[str] = []

        for tool, result in zip(self.tools, results, strict=True):
            if isinstance(result, Exception):
                errors.append(f"{tool.name}: {result!s}")
                logger.warning("Search tool failed", tool=tool.name, error=str(result))
            else:
                # Cast result to list[Evidence] as we know it succeeded
                success_result = cast(list[Evidence], result)
                all_evidence.extend(success_result)

                # Cast tool.name to the expected Literal
                tool_name = cast(Literal["pubmed"], tool.name)
                sources_searched.append(tool_name)
                logger.info("Search tool succeeded", tool=tool.name, count=len(success_result))

        return SearchResult(
            query=query,
            evidence=all_evidence,
            sources_searched=sources_searched,
            total_found=len(all_evidence),
            errors=errors,
        )

    async def _search_with_timeout(
        self,
        tool: SearchTool,
        query: str,
        max_results: int,
    ) -> list[Evidence]:
        """Execute a single tool search with timeout."""
        try:
            return await asyncio.wait_for(
                tool.search(query, max_results),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            raise SearchError(f"{tool.name} search timed out after {self.timeout}s") from e
