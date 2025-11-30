"""Tests for MCP Tools domain support."""

from unittest.mock import MagicMock, patch

from src.mcp_tools import search_pubmed


class TestMCPToolsDomain:
    @patch("src.mcp_tools._pubmed.search")
    async def test_search_pubmed_accepts_domain(self, mock_search):
        mock_search.return_value = []

        result = await search_pubmed("query", domain="sexual_health")

        # The function returns "No PubMed results found..." if empty
        assert "No PubMed results" in result

        # Let's mock results
        mock_evidence = MagicMock()
        mock_evidence.citation.title = "Test Title"
        mock_evidence.citation.authors = ["Author"]
        mock_evidence.citation.date = "2024"
        mock_evidence.citation.url = "http://url"
        mock_evidence.content = "content"

        mock_search.return_value = [mock_evidence]

        result = await search_pubmed("query", domain="sexual_health")
        assert "## PubMed Results for: query (Sexual Health Research)" in result
