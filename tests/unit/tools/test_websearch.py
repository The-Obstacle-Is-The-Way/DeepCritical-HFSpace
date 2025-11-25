"""Unit tests for WebTool."""

from unittest.mock import MagicMock

import pytest

from src.tools.websearch import WebTool


class TestWebTool:
    """Tests for WebTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """WebTool should return Evidence objects from search."""
        mock_results = [
            {
                "title": "Drug Repurposing Article",
                "href": "https://example.com/article",
                "body": "Some content about drug repurposing...",
            }
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=None)
        mock_ddgs.text = MagicMock(return_value=mock_results)

        mocker.patch("src.tools.websearch.DDGS", return_value=mock_ddgs)

        tool = WebTool()
        results = await tool.search("drug repurposing")

        assert len(results) == 1
        assert results[0].citation.source == "web"
        assert "Drug Repurposing" in results[0].citation.title
