"""Unit tests for OpenAlex tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.openalex import OpenAlexTool
from src.utils.models import Evidence

# Sample OpenAlex response
SAMPLE_OPENALEX_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W12345",
            "doi": "https://doi.org/10.1234/test",
            "display_name": "Metformin in Cancer Treatment",
            "publication_year": 2024,
            "cited_by_count": 150,
            "abstract_inverted_index": {
                "Metformin": [0],
                "shows": [1],
                "promise": [2],
                "in": [3],
                "cancer": [4],
                "treatment": [5],
            },
            "concepts": [
                {"display_name": "Metformin", "score": 0.95, "level": 2},
                {"display_name": "Cancer", "score": 0.88, "level": 1},
            ],
            "authorships": [
                {"author": {"display_name": "John Smith"}},
                {"author": {"display_name": "Jane Doe"}},
            ],
            "open_access": {"is_oa": True, "oa_url": "https://example.com/oa"},
            "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
        }
    ]
}


@pytest.mark.unit
class TestOpenAlexTool:
    """Tests for OpenAlexTool."""

    @pytest.fixture
    def tool(self) -> OpenAlexTool:
        return OpenAlexTool()

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a standardized mock client with context manager support."""
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None

        # Standard response mock
        resp = MagicMock()
        resp.json.return_value = SAMPLE_OPENALEX_RESPONSE
        resp.raise_for_status.return_value = None
        client.get.return_value = resp

        mocker.patch("httpx.AsyncClient", return_value=client)
        return client

    def test_tool_name(self, tool: OpenAlexTool) -> None:
        """Tool name should be 'openalex'."""
        assert tool.name == "openalex"

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool: OpenAlexTool, mock_client) -> None:
        """Search should return Evidence objects."""
        results = await tool.search("metformin cancer", max_results=5)

        assert len(results) == 1
        assert isinstance(results[0], Evidence)
        assert results[0].citation.source == "openalex"

    @pytest.mark.asyncio
    async def test_search_includes_citation_count(self, tool: OpenAlexTool, mock_client) -> None:
        """Evidence metadata should include cited_by_count."""
        results = await tool.search("metformin cancer", max_results=5)
        assert results[0].metadata["cited_by_count"] == 150

    @pytest.mark.asyncio
    async def test_search_calculates_relevance(self, tool: OpenAlexTool, mock_client) -> None:
        """Evidence relevance should be based on citations (capped at 1.0)."""
        results = await tool.search("metformin cancer", max_results=5)
        # 150 citations / 100 = 1.5 -> capped at 1.0
        assert results[0].relevance == 1.0

    @pytest.mark.asyncio
    async def test_search_includes_concepts(self, tool: OpenAlexTool, mock_client) -> None:
        """Evidence metadata should include concepts."""
        results = await tool.search("metformin cancer", max_results=5)
        assert "Metformin" in results[0].metadata["concepts"]
        assert "Cancer" in results[0].metadata["concepts"]

    @pytest.mark.asyncio
    async def test_search_includes_open_access_info(self, tool: OpenAlexTool, mock_client) -> None:
        """Evidence metadata should include open access info."""
        results = await tool.search("metformin cancer", max_results=5)
        assert results[0].metadata["is_open_access"] is True
        assert results[0].metadata["pdf_url"] == "https://example.com/paper.pdf"

    def test_reconstruct_abstract(self, tool: OpenAlexTool) -> None:
        """Abstract reconstruction from inverted index."""
        inverted_index = {
            "Hello": [0],
            "world": [1],
            "this": [2],
            "is": [3],
            "a": [4],
            "test": [5],
        }
        result = tool._reconstruct_abstract(inverted_index)
        assert result == "Hello world this is a test"

    def test_reconstruct_abstract_empty(self, tool: OpenAlexTool) -> None:
        """Handle None or empty inverted index."""
        assert tool._reconstruct_abstract(None) == ""
        assert tool._reconstruct_abstract({}) == ""

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool: OpenAlexTool, mock_client) -> None:
        """Handle empty results gracefully."""
        mock_client.get.return_value.json.return_value = {"results": []}

        results = await tool.search("xyznonexistent123", max_results=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_params(self, tool: OpenAlexTool, mock_client) -> None:
        """Verify API call requests citation-sorted results and uses polite pool."""
        mock_client.get.return_value.json.return_value = {"results": []}

        await tool.search("test query", max_results=5)

        # Verify call params
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["sort"] == "cited_by_count:desc"
        assert params["mailto"] == tool.POLITE_EMAIL
        assert "type:article" in params["filter"]
        assert "has_abstract:true" in params["filter"]


@pytest.mark.integration
class TestOpenAlexIntegration:
    """Integration tests with real OpenAlex API."""

    @pytest.mark.asyncio
    async def test_real_api_returns_results(self) -> None:
        """Test actual API returns relevant results."""
        tool = OpenAlexTool()
        results = await tool.search("metformin cancer treatment", max_results=3)

        assert len(results) > 0
        # Should have citation counts
        assert results[0].metadata["cited_by_count"] >= 0
        # Should have abstract text
        assert len(results[0].content) > 50
        # Should have concepts
        assert len(results[0].metadata["concepts"]) > 0
