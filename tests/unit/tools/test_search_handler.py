"""Unit tests for SearchHandler."""

from unittest.mock import AsyncMock

import pytest

from src.tools.search_handler import SearchHandler
from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class TestSearchHandler:
    """Tests for SearchHandler."""

    @pytest.mark.asyncio
    async def test_execute_aggregates_results(self):
        """SearchHandler should aggregate results from all tools."""
        # Create mock tools
        mock_tool_1 = AsyncMock()
        mock_tool_1.name = "pubmed"
        mock_tool_1.search = AsyncMock(
            return_value=[
                Evidence(
                    content="Result 1",
                    citation=Citation(source="pubmed", title="T1", url="u1", date="2024"),
                )
            ]
        )

        mock_tool_2 = AsyncMock()
        mock_tool_2.name = "pubmed"  # Type system currently restricts to pubmed
        mock_tool_2.search = AsyncMock(return_value=[])

        handler = SearchHandler(tools=[mock_tool_1, mock_tool_2])
        result = await handler.execute("test query")

        assert result.total_found == 1
        assert "pubmed" in result.sources_searched
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_handles_tool_failure(self):
        """SearchHandler should continue if one tool fails."""
        mock_tool_ok = AsyncMock()
        mock_tool_ok.name = "pubmed"
        mock_tool_ok.search = AsyncMock(
            return_value=[
                Evidence(
                    content="Good result",
                    citation=Citation(source="pubmed", title="T", url="u", date="2024"),
                )
            ]
        )

        mock_tool_fail = AsyncMock()
        mock_tool_fail.name = "pubmed"  # Mocking a second pubmed instance failing
        mock_tool_fail.search = AsyncMock(side_effect=SearchError("API down"))

        handler = SearchHandler(tools=[mock_tool_ok, mock_tool_fail])
        result = await handler.execute("test")

        assert result.total_found == 1
        assert "pubmed" in result.sources_searched
        assert len(result.errors) == 1
        # The error message format is "{tool.name}: {error!s}"
        assert "pubmed: API down" in result.errors[0]

    @pytest.mark.asyncio
    async def test_search_handler_pubmed_only(self):
        """SearchHandler should work with only PubMed tool."""
        # This is the specific test requested in Phase 9 spec
        from src.tools.pubmed import PubMedTool

        mock_pubmed = AsyncMock(spec=PubMedTool)
        mock_pubmed.name = "pubmed"
        mock_pubmed.search.return_value = []

        handler = SearchHandler(tools=[mock_pubmed], timeout=30.0)
        result = await handler.execute("metformin diabetes", max_results_per_tool=3)

        assert result.sources_searched == ["pubmed"]
        assert "web" not in result.sources_searched
        assert len(result.errors) == 0
