"""Unit tests for Europe PMC tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.europepmc import EuropePMCTool
from src.utils.models import Evidence


@pytest.mark.unit
class TestEuropePMCTool:
    """Tests for EuropePMCTool."""

    @pytest.fixture
    def tool(self) -> EuropePMCTool:
        return EuropePMCTool()

    def test_tool_name(self, tool: EuropePMCTool) -> None:
        assert tool.name == "europepmc"

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool: EuropePMCTool) -> None:
        """Test that search returns Evidence objects."""
        mock_response = {
            "resultList": {
                "result": [
                    {
                        "id": "12345",
                        "title": "Testosterone Therapy for HSDD Study",
                        "abstractText": "This study examines testosterone therapy for HSDD.",
                        "doi": "10.1234/test",
                        "pubYear": "2024",
                        "source": "MED",
                        "pubTypeList": {"pubType": ["research-article"]},
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            # Create response mock
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status.return_value = None

            mock_instance.get.return_value = mock_resp

            results = await tool.search("testosterone HSDD therapy", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert "Testosterone Therapy for HSDD Study" in results[0].citation.title

    @pytest.mark.asyncio
    async def test_search_marks_preprints(self, tool: EuropePMCTool) -> None:
        """Test that preprints are marked correctly."""
        mock_response = {
            "resultList": {
                "result": [
                    {
                        "id": "PPR12345",
                        "title": "Preprint Study",
                        "abstractText": "Abstract text",
                        "doi": "10.1234/preprint",
                        "pubYear": "2024",
                        "source": "PPR",
                        "pubTypeList": {"pubType": ["Preprint"]},
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("test", max_results=5)

            assert "PREPRINT" in results[0].content
            assert results[0].citation.source == "preprint"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool: EuropePMCTool) -> None:
        """Test handling of empty results."""
        mock_response = {"resultList": {"result": []}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_resp

            results = await tool.search("nonexistent query xyz", max_results=5)

            assert results == []


@pytest.mark.integration
class TestEuropePMCIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_api_call(self) -> None:
        """Test actual API returns relevant results for sexual health query."""
        tool = EuropePMCTool()
        results = await tool.search("testosterone libido therapy", max_results=3)

        assert len(results) > 0
        # At least one result should mention testosterone or libido
        titles = " ".join([r.citation.title.lower() for r in results])
        assert "testosterone" in titles or "libido" in titles or "sexual" in titles
