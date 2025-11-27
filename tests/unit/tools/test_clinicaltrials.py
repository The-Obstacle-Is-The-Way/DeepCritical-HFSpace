"""Unit tests for ClinicalTrials.gov tool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.clinicaltrials import ClinicalTrialsTool
from src.utils.models import Evidence


@pytest.mark.unit
class TestClinicalTrialsTool:
    """Tests for ClinicalTrialsTool."""

    @pytest.fixture
    def tool(self) -> ClinicalTrialsTool:
        return ClinicalTrialsTool()

    def test_tool_name(self, tool: ClinicalTrialsTool) -> None:
        assert tool.name == "clinicaltrials"

    @pytest.mark.asyncio
    async def test_search_uses_filters(self, tool: ClinicalTrialsTool) -> None:
        """Test that search applies status and type filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            await tool.search("test query", max_results=5)

            # Verify filters were applied
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", call_args[1].get("params", {}))

            # Should filter for active/completed studies
            assert "filter.overallStatus" in params
            assert "COMPLETED" in params["filter.overallStatus"]
            assert "RECRUITING" in params["filter.overallStatus"]

            # Should filter for interventional studies via query term
            assert "AREA[StudyType]INTERVENTIONAL" in params["query.term"]
            assert "filter.studyType" not in params

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool: ClinicalTrialsTool) -> None:
        """Test that search returns Evidence objects."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Metformin for Long COVID Treatment",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2023-01-01"},
                },
                "descriptionModule": {
                    "briefSummary": "A study examining metformin for Long COVID symptoms.",
                },
                "designModule": {
                    "phases": ["PHASE2", "PHASE3"],
                },
                "conditionsModule": {
                    "conditions": ["Long COVID", "PASC"],
                },
                "armsInterventionsModule": {
                    "interventions": [{"name": "Metformin"}],
                },
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("long covid metformin", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert "Metformin" in results[0].citation.title
            assert "PHASE2" in results[0].content or "Phase" in results[0].content

    @pytest.mark.asyncio
    async def test_search_includes_phase_info(self, tool: ClinicalTrialsTool) -> None:
        """Test that phase information is included in content."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-01-01"},
                },
                "descriptionModule": {
                    "briefSummary": "Test summary.",
                },
                "designModule": {
                    "phases": ["PHASE3"],
                },
                "conditionsModule": {"conditions": ["Test"]},
                "armsInterventionsModule": {"interventions": []},
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=5)

            # Phase should be in content
            assert "PHASE3" in results[0].content or "Phase 3" in results[0].content

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool: ClinicalTrialsTool) -> None:
        """Test handling of empty results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("nonexistent xyz 12345", max_results=5)
            assert results == []


@pytest.mark.integration
class TestClinicalTrialsIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_api_returns_interventional(self) -> None:
        """Test that real API returns interventional studies."""
        tool = ClinicalTrialsTool()
        results = await tool.search("long covid treatment", max_results=3)

        # Should get results
        assert len(results) > 0

        # Results should mention interventions or treatments
        all_content = " ".join([r.content.lower() for r in results])
        has_intervention = (
            "intervention" in all_content
            or "treatment" in all_content
            or "drug" in all_content
            or "phase" in all_content
        )
        assert has_intervention
