"""Unit tests for Orchestrator Factory."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from src.orchestrator import Orchestrator
from src.orchestrator_factory import create_orchestrator


@pytest.fixture
def mock_settings():
    with patch("src.orchestrator_factory.settings", autospec=True) as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_magentic_cls():
    with patch("src.orchestrator_factory._get_magentic_orchestrator_class") as mock:
        # The mock returns a class (callable), which returns an instance
        mock_class = MagicMock()
        mock.return_value = mock_class
        yield mock_class


@pytest.fixture
def mock_handlers():
    return MagicMock(), MagicMock()


def test_create_orchestrator_simple_explicit(mock_settings, mock_handlers):
    """Test explicit simple mode."""
    search, judge = mock_handlers
    orch = create_orchestrator(search_handler=search, judge_handler=judge, mode="simple")
    assert isinstance(orch, Orchestrator)


def test_create_orchestrator_advanced_explicit(mock_settings, mock_handlers, mock_magentic_cls):
    """Test explicit advanced mode."""
    # Ensure has_openai_key is True so it doesn't error if we add checks
    mock_settings.has_openai_key = True

    orch = create_orchestrator(mode="advanced")
    # verify instantiated
    mock_magentic_cls.assert_called_once()
    assert orch == mock_magentic_cls.return_value


def test_create_orchestrator_auto_advanced(mock_settings, mock_magentic_cls):
    """Test auto-detect advanced mode when OpenAI key exists."""
    mock_settings.has_openai_key = True

    orch = create_orchestrator()
    mock_magentic_cls.assert_called_once()
    assert orch == mock_magentic_cls.return_value


def test_create_orchestrator_auto_simple(mock_settings, mock_handlers):
    """Test auto-detect simple mode when no paid keys."""
    mock_settings.has_openai_key = False

    search, judge = mock_handlers
    orch = create_orchestrator(search_handler=search, judge_handler=judge)
    assert isinstance(orch, Orchestrator)
