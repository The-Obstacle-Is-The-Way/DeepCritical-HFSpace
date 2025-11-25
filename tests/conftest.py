"""Shared pytest fixtures for all tests."""

from unittest.mock import AsyncMock

import pytest


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


# NOTE: sample_evidence fixture will be added in Phase 2 when models.py exists
# @pytest.fixture
# def sample_evidence():
#     """Sample Evidence objects for testing."""
#     from src.utils.models import Citation, Evidence
#     return [...]
