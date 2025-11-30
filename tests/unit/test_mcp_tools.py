"""Unit tests for MCP tool wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools import (
    search_all_sources,
    search_clinical_trials,
    search_europepmc,
    search_pubmed,
)
from src.utils.models import Citation, Evidence


@pytest.fixture
def mock_evidence() -> Evidence:
    """Sample evidence for testing."""
    return Evidence(
        content="Testosterone therapy shows efficacy in treating HSDD.",
        citation=Citation(
            source="pubmed",
            title="Testosterone and Female Libido",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            date="2024-01-15",
            authors=["Smith J", "Jones M", "Brown K"],
        ),
        relevance=0.85,
    )


class TestSearchPubMed:
    """Tests for search_pubmed MCP tool."""

    @pytest.mark.asyncio
    @patch("src.mcp_tools._pubmed.search")
    async def test_returns_formatted_string(self, mock_search):
        """Test that search_pubmed returns Markdown formatted string."""
        # Mock evidence
        mock_evidence = MagicMock()
        mock_evidence.citation.title = "Test Title"
        mock_evidence.citation.authors = ["Author 1", "Author 2"]
        mock_evidence.citation.date = "2024"
        mock_evidence.citation.url = "http://test.com"
        mock_evidence.content = "Abstract content..."

        mock_search.return_value = [mock_evidence]

        with patch("src.mcp_tools.get_domain_config") as mock_config:
            mock_config.return_value.name = "Sexual Health Research"

            result = await search_pubmed("testosterone libido", 10)

            assert "## PubMed Results" in result
            assert "Sexual Health Research" in result
            assert "Test Title" in result
            assert "Author 1" in result
            assert "2024" in result
            assert "Abstract content..." in result

    @pytest.mark.asyncio
    async def test_clamps_max_results(self) -> None:
        """Should clamp max_results to valid range (1-50)."""
        with patch("src.mcp_tools._pubmed") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[])

            # Test lower bound
            await search_pubmed("test", 0)
            mock_tool.search.assert_called_with("test", 1)

            # Test upper bound
            await search_pubmed("test", 100)
            mock_tool.search.assert_called_with("test", 50)

    @pytest.mark.asyncio
    async def test_handles_no_results(self) -> None:
        """Should return appropriate message when no results."""
        with patch("src.mcp_tools._pubmed") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[])

            result = await search_pubmed("xyznonexistent", 10)

            assert "No PubMed results found" in result


class TestSearchClinicalTrials:
    """Tests for search_clinical_trials MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, mock_evidence: Evidence) -> None:
        """Should return formatted markdown string."""
        mock_evidence.citation.source = "clinicaltrials"  # type: ignore

        with patch("src.mcp_tools._trials") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[mock_evidence])

            result = await search_clinical_trials("sildenafil erectile dysfunction", 10)

            assert isinstance(result, str)
            assert "Clinical Trials" in result


class TestSearchEuropePMC:
    """Tests for search_europepmc MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, mock_evidence: Evidence) -> None:
        """Should return formatted markdown string."""
        mock_evidence.citation.source = "europepmc"  # type: ignore

        with patch("src.mcp_tools._europepmc") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[mock_evidence])

            result = await search_europepmc("preprint search", 10)

            assert isinstance(result, str)
            assert "Europe PMC Results" in result


class TestSearchAllSources:
    """Tests for search_all_sources MCP tool."""

    @pytest.mark.asyncio
    async def test_combines_all_sources(self, mock_evidence: Evidence) -> None:
        """Should combine results from all sources."""
        with (
            patch("src.mcp_tools.search_pubmed", new_callable=AsyncMock) as mock_pubmed,
            patch("src.mcp_tools.search_clinical_trials", new_callable=AsyncMock) as mock_trials,
            patch("src.mcp_tools.search_europepmc", new_callable=AsyncMock) as mock_europepmc,
        ):
            mock_pubmed.return_value = "## PubMed Results"
            mock_trials.return_value = "## Clinical Trials"
            mock_europepmc.return_value = "## Europe PMC Results"

            result = await search_all_sources("testosterone libido", 5)

            assert "Comprehensive Search" in result
            assert "PubMed" in result
            assert "Clinical Trials" in result
            assert "Europe PMC" in result

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self) -> None:
        """Should handle partial failures gracefully."""
        with (
            patch("src.mcp_tools.search_pubmed", new_callable=AsyncMock) as mock_pubmed,
            patch("src.mcp_tools.search_clinical_trials", new_callable=AsyncMock) as mock_trials,
            patch("src.mcp_tools.search_europepmc", new_callable=AsyncMock) as mock_europepmc,
        ):
            mock_pubmed.return_value = "## PubMed Results"
            mock_trials.side_effect = Exception("API Error")
            mock_europepmc.return_value = "## Europe PMC Results"

            result = await search_all_sources("testosterone libido", 5)

            # Should still contain working sources
            assert "PubMed" in result
            assert "Europe PMC" in result
            # Should show error for failed source
            assert "Error" in result


class TestMCPDocstrings:
    """Tests that docstrings follow MCP format."""

    def test_search_pubmed_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_pubmed.__doc__ is not None
        assert "Args:" in search_pubmed.__doc__
        assert "query:" in search_pubmed.__doc__
        assert "max_results:" in search_pubmed.__doc__
        assert "Returns:" in search_pubmed.__doc__

    def test_search_clinical_trials_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_clinical_trials.__doc__ is not None
        assert "Args:" in search_clinical_trials.__doc__

    def test_search_europepmc_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_europepmc.__doc__ is not None
        assert "Args:" in search_europepmc.__doc__

    def test_search_all_sources_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_all_sources.__doc__ is not None
        assert "Args:" in search_all_sources.__doc__


class TestMCPTypeHints:
    """Tests that type hints are complete for MCP."""

    def test_search_pubmed_type_hints(self) -> None:
        """All parameters and return must have type hints."""
        import inspect

        sig = inspect.signature(search_pubmed)

        # Check parameter hints
        assert sig.parameters["query"].annotation is str
        assert sig.parameters["max_results"].annotation is int

        # Check return hint
        assert sig.return_annotation is str

    def test_search_clinical_trials_type_hints(self) -> None:
        """All parameters and return must have type hints."""
        import inspect

        sig = inspect.signature(search_clinical_trials)
        assert sig.parameters["query"].annotation is str
        assert sig.parameters["max_results"].annotation is int
        assert sig.return_annotation is str
