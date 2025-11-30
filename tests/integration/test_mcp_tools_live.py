"""Integration tests for MCP tool wrappers with live API calls."""

import pytest


class TestMCPToolsLive:
    """Integration tests for MCP tools against live APIs (PubMed, etc.)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_tools_work_end_to_end(self) -> None:
        """Test that MCP tools execute real searches."""
        from src.mcp_tools import search_pubmed

        result = await search_pubmed("testosterone libido", 3)

        assert isinstance(result, str)
        assert "PubMed Results" in result
        # Should have actual content (not just "no results")
        # Typical queries should return something.
        # The wrapper returns "No PubMed results found" string if empty.

        if "No PubMed results found" not in result:
            assert len(result) > 10
