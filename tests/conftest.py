"""Shared pytest fixtures for all tests."""

from unittest.mock import AsyncMock

import pytest

from src.utils.models import Citation, Evidence


@pytest.fixture
def mock_httpx_client(mocker):
    """Mock httpx.AsyncClient for API tests."""
    mock = mocker.patch("httpx.AsyncClient")
    mock.return_value.__aenter__ = AsyncMock(return_value=mock.return_value)
    mock.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_llm_response():
    """Factory fixture for mocking LLM responses."""

    def _mock(content: str):
        return AsyncMock(return_value=content)

    return _mock


@pytest.fixture
def sample_evidence():
    """Sample Evidence objects for testing."""
    return [
        Evidence(
            content="Testosterone shows efficacy in treating hypoactive sexual desire disorder...",
            citation=Citation(
                source="pubmed",
                title="Testosterone and Female Libido: A Systematic Review",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                date="2024-01-15",
                authors=["Smith J", "Johnson M"],
            ),
            relevance=0.85,
        ),
        Evidence(
            content="Transdermal testosterone offers effective treatment path...",
            citation=Citation(
                source="pubmed",
                title="Testosterone Therapy Strategies",
                url="https://example.com/testosterone-therapy",
                date="Unknown",
                authors=[],
            ),
            relevance=0.72,
        ),
    ]
