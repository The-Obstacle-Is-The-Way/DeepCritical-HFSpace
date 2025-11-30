"""Unit tests for SearchHandler."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.tools.base import SearchTool
from src.tools.search_handler import SearchHandler, deduplicate_evidence, extract_paper_id
from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


def _make_evidence(source: str, url: str, metadata: dict | None = None) -> Evidence:
    """Helper to create Evidence objects for testing."""
    return Evidence(
        content="Test content",
        citation=Citation(
            source=source,
            title="Test",
            url=url,
            date="2024",
            authors=[],
        ),
        metadata=metadata or {},
    )


class TestExtractPaperId:
    """Tests for paper ID extraction from Evidence objects."""

    def test_extracts_pubmed_id(self) -> None:
        evidence = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        assert extract_paper_id(evidence) == "PMID:12345678"

    def test_extracts_europepmc_med_id(self) -> None:
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/MED/12345678")
        assert extract_paper_id(evidence) == "PMID:12345678"

    def test_extracts_europepmc_pmc_id(self) -> None:
        """Europe PMC PMC articles have different ID format."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PMC/PMC7654321")
        assert extract_paper_id(evidence) == "PMCID:PMC7654321"

    def test_extracts_europepmc_ppr_id(self) -> None:
        """Europe PMC preprints have PPR IDs."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PPR/PPR123456")
        assert extract_paper_id(evidence) == "PPRID:PPR123456"

    def test_extracts_europepmc_pat_id(self) -> None:
        """Europe PMC patents have PAT IDs (WIPO format)."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PAT/WO8601415")
        assert extract_paper_id(evidence) == "PATID:WO8601415"

    def test_extracts_europepmc_pat_id_eu_format(self) -> None:
        """European patent format should also work."""
        evidence = _make_evidence("europepmc", "https://europepmc.org/article/PAT/EP1234567")
        assert extract_paper_id(evidence) == "PATID:EP1234567"

    def test_extracts_doi(self) -> None:
        evidence = _make_evidence("pubmed", "https://doi.org/10.1038/nature12345")
        assert extract_paper_id(evidence) == "DOI:10.1038/nature12345"

    def test_extracts_doi_with_trailing_slash(self) -> None:
        """DOIs should be normalized (trailing slash removed)."""
        evidence = _make_evidence("pubmed", "https://doi.org/10.1038/nature12345/")
        assert extract_paper_id(evidence) == "DOI:10.1038/nature12345"

    def test_extracts_openalex_id_from_url(self) -> None:
        """OpenAlex ID from URL (fallback when no PMID in metadata)."""
        evidence = _make_evidence("openalex", "https://openalex.org/W1234567890")
        assert extract_paper_id(evidence) == "OAID:W1234567890"

    def test_extracts_openalex_pmid_from_metadata(self) -> None:
        """OpenAlex PMID from metadata takes priority over URL."""
        evidence = _make_evidence(
            "openalex",
            "https://openalex.org/W1234567890",
            metadata={"pmid": "98765432"},
        )
        assert extract_paper_id(evidence) == "PMID:98765432"

    def test_extracts_nct_id_modern(self) -> None:
        evidence = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT12345678")
        assert extract_paper_id(evidence) == "NCT:NCT12345678"

    def test_extracts_nct_id_legacy(self) -> None:
        """Legacy ClinicalTrials.gov URL format should also work."""
        evidence = _make_evidence(
            "clinicaltrials", "https://clinicaltrials.gov/ct2/show/NCT12345678"
        )
        assert extract_paper_id(evidence) == "NCT:NCT12345678"

    def test_returns_none_for_unknown_url(self) -> None:
        evidence = _make_evidence("web", "https://example.com/unknown")
        assert extract_paper_id(evidence) is None


class TestDeduplicateEvidence:
    """Tests for evidence deduplication."""

    def test_removes_pubmed_europepmc_duplicate(self) -> None:
        """Same paper from PubMed and Europe PMC should dedupe to PubMed."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        europepmc = _make_evidence("europepmc", "https://europepmc.org/article/MED/12345678")

        result = deduplicate_evidence([pubmed, europepmc])

        assert len(result) == 1
        assert result[0].citation.source == "pubmed"

    def test_removes_pubmed_openalex_duplicate_via_metadata(self) -> None:
        """OpenAlex with PMID in metadata should dedupe against PubMed."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        openalex = _make_evidence(
            "openalex",
            "https://openalex.org/W9999999",
            metadata={"pmid": "12345678", "cited_by_count": 100},
        )

        result = deduplicate_evidence([pubmed, openalex])

        assert len(result) == 1
        assert result[0].citation.source == "pubmed"

    def test_preserves_unique_evidence(self) -> None:
        """Different papers should not be deduplicated."""
        e1 = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/11111111/")
        e2 = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/22222222/")

        result = deduplicate_evidence([e1, e2])

        assert len(result) == 2

    def test_preserves_openalex_without_pmid(self) -> None:
        """OpenAlex papers without PMID should NOT be deduplicated against PubMed."""
        pubmed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        openalex_no_pmid = _make_evidence(
            "openalex",
            "https://openalex.org/W9999999",
            metadata={"cited_by_count": 100},  # No pmid key
        )

        result = deduplicate_evidence([pubmed, openalex_no_pmid])

        assert len(result) == 2  # Both preserved (different IDs)

    def test_keeps_unidentifiable_evidence(self) -> None:
        """Evidence with unrecognized URLs should be preserved."""
        unknown = _make_evidence("web", "https://example.com/paper/123")

        result = deduplicate_evidence([unknown])

        assert len(result) == 1

    def test_clinicaltrials_unique_per_nct(self) -> None:
        """ClinicalTrials entries have unique NCT IDs."""
        trial1 = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT11111111")
        trial2 = _make_evidence("clinicaltrials", "https://clinicaltrials.gov/study/NCT22222222")

        result = deduplicate_evidence([trial1, trial2])

        assert len(result) == 2

    def test_preprints_preserved_separately(self) -> None:
        """Preprints (PPR IDs) should not dedupe against peer-reviewed papers."""
        peer_reviewed = _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        preprint = _make_evidence("europepmc", "https://europepmc.org/article/PPR/PPR999999")

        result = deduplicate_evidence([peer_reviewed, preprint])

        assert len(result) == 2  # Both preserved (different ID types)


class TestSearchHandler:
    """Tests for SearchHandler."""

    @pytest.mark.asyncio
    async def test_execute_aggregates_and_deduplicates(self):
        """SearchHandler should aggregate results and deduplicate them."""
        # Setup
        mock_tool1 = AsyncMock(spec=SearchTool)
        mock_tool1.name = "pubmed"
        mock_tool1.search.return_value = [
            _make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        ]

        mock_tool2 = AsyncMock(spec=SearchTool)
        mock_tool2.name = "europepmc"
        # Duplicate of the pubmed result
        mock_tool2.search.return_value = [
            _make_evidence("europepmc", "https://europepmc.org/article/MED/12345678")
        ]

        handler = SearchHandler(tools=[mock_tool1, mock_tool2])

        # Execute
        result = await handler.execute("test")

        # Should only have 1 result after deduplication
        assert result.total_found == 1
        assert len(result.evidence) == 1
        assert result.evidence[0].citation.source == "pubmed"  # Priority source kept
        assert "pubmed" in result.sources_searched
        assert "europepmc" in result.sources_searched

    @pytest.mark.asyncio
    async def test_execute_handles_tool_failure(self):
        """SearchHandler should continue if one tool fails."""
        mock_tool_ok = create_autospec(SearchTool, instance=True)
        mock_tool_ok.name = "pubmed"
        mock_tool_ok.search = AsyncMock(
            return_value=[_make_evidence("pubmed", "https://pubmed.ncbi.nlm.nih.gov/12345678/")]
        )

        mock_tool_fail = create_autospec(SearchTool, instance=True)
        mock_tool_fail.name = "clinicaltrials"
        mock_tool_fail.search = AsyncMock(side_effect=SearchError("API down"))

        handler = SearchHandler(tools=[mock_tool_ok, mock_tool_fail])
        result = await handler.execute("test")

        assert result.total_found == 1
        assert "pubmed" in result.sources_searched
        assert len(result.errors) == 1
        assert "clinicaltrials: API down" in result.errors[0]
