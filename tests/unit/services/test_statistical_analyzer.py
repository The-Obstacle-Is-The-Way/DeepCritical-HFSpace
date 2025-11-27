"Unit tests for StatisticalAnalyzer service."

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.statistical_analyzer import (
    AnalysisResult,
    StatisticalAnalyzer,
    get_statistical_analyzer,
)
from src.utils.models import Citation, Evidence


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Sample evidence for testing."""
    return [
        Evidence(
            content="Metformin shows effect size of 0.45.",
            citation=Citation(
                source="pubmed",
                title="Metformin Study",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                date="2024-01-15",
                authors=["Smith J"],
            ),
            relevance=0.9,
        )
    ]


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer (no agent_framework dependency)."""

    def test_no_agent_framework_import(self) -> None:
        """StatisticalAnalyzer must NOT import agent_framework."""
        import src.services.statistical_analyzer as module

        # Check module doesn't import agent_framework
        with open(module.__file__) as f:
            source = f.read()
        assert "from agent_framework" not in source
        assert "import agent_framework" not in source
        assert "BaseAgent" not in source

    @pytest.mark.asyncio
    async def test_analyze_returns_result(self, sample_evidence: list[Evidence]) -> None:
        """analyze() should return AnalysisResult."""
        analyzer = StatisticalAnalyzer()

        with (
            patch.object(analyzer, "_get_agent") as mock_agent,
            patch.object(analyzer, "_get_code_executor") as mock_executor,
        ):
            # Mock LLM
            mock_agent.return_value.run = AsyncMock(
                return_value=MagicMock(output="print('SUPPORTED')")
            )

            # Mock Modal
            mock_executor.return_value.execute.return_value = {
                "stdout": "SUPPORTED\np-value: 0.01",
                "stderr": "",
                "success": True,
            }

            result = await analyzer.analyze("test query", sample_evidence)

            assert isinstance(result, AnalysisResult)
            assert result.verdict == "SUPPORTED"

    def test_singleton(self) -> None:
        """get_statistical_analyzer should return singleton."""
        a1 = get_statistical_analyzer()
        a2 = get_statistical_analyzer()
        assert a1 is a2


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_verdict_values(self) -> None:
        """Verdict should be one of the expected values."""
        for verdict in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]:
            result = AnalysisResult(
                verdict=verdict,  # type: ignore
                confidence=0.8,
                statistical_evidence="test",
                code_generated="print('test')",
                execution_output="test",
            )
            assert result.verdict == verdict

    def test_confidence_bounds(self) -> None:
        """Confidence must be 0.0-1.0."""
        with pytest.raises(ValueError):
            AnalysisResult(
                verdict="SUPPORTED",
                confidence=1.5,  # Invalid
                statistical_evidence="test",
                code_generated="test",
                execution_output="test",
            )
