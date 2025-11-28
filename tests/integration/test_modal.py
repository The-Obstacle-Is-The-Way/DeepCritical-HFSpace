"""Integration tests for Modal (requires credentials and modal package)."""

import pytest

from src.utils.config import settings

# Check if any LLM API key is available
_llm_available = bool(settings.openai_api_key or settings.anthropic_api_key)

# Check if modal package is installed
try:
    import modal  # noqa: F401

    _modal_installed = True
except ImportError:
    _modal_installed = False


@pytest.mark.integration
@pytest.mark.skipif(not _modal_installed, reason="Modal package not installed")
@pytest.mark.skipif(not settings.modal_available, reason="Modal credentials not configured")
class TestModalIntegration:
    """Integration tests requiring Modal credentials."""

    @pytest.mark.asyncio
    async def test_sandbox_executes_code(self) -> None:
        """Modal sandbox should execute Python code."""
        import asyncio
        from functools import partial

        from src.tools.code_execution import get_code_executor

        executor = get_code_executor()
        code = "import pandas as pd; print(pd.DataFrame({'a': [1,2,3]})['a'].sum())"

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(executor.execute, code, timeout=30))

        assert result["success"]
        assert "6" in result["stdout"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _llm_available, reason="LLM API key not configured")
    async def test_statistical_analyzer_works(self) -> None:
        """StatisticalAnalyzer should work end-to-end (requires Modal + LLM)."""
        from src.services.statistical_analyzer import get_statistical_analyzer
        from src.utils.models import Citation, Evidence

        evidence = [
            Evidence(
                content="Drug shows 40% improvement in trial.",
                citation=Citation(
                    source="pubmed",
                    title="Test",
                    url="https://test.com",
                    date="2024-01-01",
                    authors=["Test"],
                ),
                relevance=0.9,
            )
        ]

        analyzer = get_statistical_analyzer()
        result = await analyzer.analyze("test drug efficacy", evidence)

        assert result.verdict in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]
        assert 0.0 <= result.confidence <= 1.0
