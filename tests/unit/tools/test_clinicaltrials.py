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
                    "briefTitle": "Testosterone for HSDD Treatment",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2023-01-01"},
                },
                "descriptionModule": {
                    "briefSummary": "A study examining testosterone for HSDD symptoms.",
                },
                "designModule": {
                    "phases": ["PHASE2", "PHASE3"],
                },
                "conditionsModule": {
                    "conditions": ["HSDD", "Hypoactive Sexual Desire"],
                },
                "armsInterventionsModule": {
                    "interventions": [{"name": "Testosterone"}],
                },
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("testosterone hsdd", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert "Testosterone" in results[0].citation.title
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


@pytest.mark.unit
class TestClinicalTrialsOutcomes:
    """Tests for outcome measure extraction."""

    @pytest.fixture
    def tool(self) -> ClinicalTrialsTool:
        return ClinicalTrialsTool()

    @pytest.mark.asyncio
    async def test_extracts_primary_outcome(self, tool: ClinicalTrialsTool) -> None:
        """Test that primary outcome is extracted from response."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2023"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE3"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Change in IIEF-EF score", "timeFrame": "Week 12"}
                    ]
                },
            },
            "hasResults": True,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert len(results) == 1
            assert "Primary Outcome" in results[0].content
            assert "IIEF-EF" in results[0].content
            assert "Week 12" in results[0].content

    @pytest.mark.asyncio
    async def test_includes_results_status(self, tool: ClinicalTrialsTool) -> None:
        """Test that results availability is shown."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2023"},
                    # Note: resultsFirstPostDateStruct, not resultsFirstSubmitDate
                    "resultsFirstPostDateStruct": {"date": "2024-06-15"},
                },
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE3"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": True,  # Note: hasResults is TOP-LEVEL
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert "Results Available: Yes" in results[0].content
            assert "2024-06-15" in results[0].content

    @pytest.mark.asyncio
    async def test_shows_no_results_when_missing(self, tool: ClinicalTrialsTool) -> None:
        """Test that missing results are indicated."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT99999999",
                    "briefTitle": "Test Study",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024"},
                },
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE2"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert "Results Available: Not yet posted" in results[0].content

    @pytest.mark.asyncio
    async def test_boosts_relevance_for_results(self, tool: ClinicalTrialsTool) -> None:
        """Trials with results should have higher relevance score."""
        with_results = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT11111111", "briefTitle": "With Results"},
                "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2023"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": []},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": True,
        }
        without_results = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT22222222", "briefTitle": "No Results"},
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024"},
                },
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": []},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [with_results, without_results]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=2)

            assert results[0].relevance == 0.90  # With results
            assert results[1].relevance == 0.85  # Without results


@pytest.mark.integration
class TestClinicalTrialsIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_api_returns_interventional(self) -> None:
        """Test that real API returns interventional studies for sexual health query."""
        tool = ClinicalTrialsTool()
        results = await tool.search("testosterone HSDD", max_results=3)

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

    @pytest.mark.asyncio
    async def test_real_completed_trial_has_outcome(self) -> None:
        """Real completed Phase 3 trials should have outcome measures."""
        tool = ClinicalTrialsTool()

        # Search for completed Phase 3 ED trials (likely to have outcomes)
        results = await tool.search(
            "sildenafil erectile dysfunction Phase 3 COMPLETED", max_results=3
        )

        # Skip if API returns no results (external dependency)
        if not results:
            pytest.skip("API returned no results for this query")

        # At least one should have primary outcome
        has_outcome = any("Primary Outcome" in r.content for r in results)
        assert has_outcome, "No completed trials with outcome measures found"
