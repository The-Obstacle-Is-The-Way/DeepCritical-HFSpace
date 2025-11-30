"""Unit tests for SearchHandler."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.tools.base import SearchTool
from src.tools.search_handler import SearchHandler
from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class TestSearchHandler:
    """Tests for SearchHandler."""

    @pytest.mark.asyncio
    async def test_execute_aggregates_results(self):
        """SearchHandler should aggregate results from all tools."""
        # Setup
        mock_tool1 = AsyncMock(spec=SearchTool)
        mock_tool1.name = "pubmed"
        mock_tool1.search.return_value = [
            Evidence(
                content="C1",
                citation=Citation(source="pubmed", title="T1", url="u1", date="2024"),
            )
        ]

        mock_tool2 = AsyncMock(spec=SearchTool)
        mock_tool2.name = "clinicaltrials"
        mock_tool2.search.return_value = [
            Evidence(
                content="C2",
                citation=Citation(source="clinicaltrials", title="T2", url="u2", date="2024"),
            )
        ]

        handler = SearchHandler(tools=[mock_tool1, mock_tool2])

        # Execute
        result = await handler.execute("testosterone libido", max_results_per_tool=3)
        assert result.total_found == 2
        assert "pubmed" in result.sources_searched
        assert "clinicaltrials" in result.sources_searched

    @pytest.mark.asyncio
    async def test_execute_handles_tool_failure(self):
        """SearchHandler should continue if one tool fails."""
        mock_tool_ok = create_autospec(SearchTool, instance=True)
        mock_tool_ok.name = "pubmed"
        mock_tool_ok.search = AsyncMock(
            return_value=[
                Evidence(
                    content="Good result",
                    citation=Citation(source="pubmed", title="T", url="u", date="2024"),
                )
            ]
        )

        mock_tool_fail = create_autospec(SearchTool, instance=True)
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
        result = await handler.execute("testosterone libido", max_results_per_tool=3)

        assert result.sources_searched == ["pubmed"]
        assert "web" not in result.sources_searched
        assert len(result.errors) == 0
