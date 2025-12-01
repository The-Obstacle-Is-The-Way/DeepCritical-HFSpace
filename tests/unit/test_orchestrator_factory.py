"""Unit tests for Orchestrator Factory."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from src.orchestrators import create_orchestrator


@pytest.fixture
def mock_settings():
    with patch("src.orchestrators.factory.settings", autospec=True) as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_advanced_cls():
    with patch("src.orchestrators.factory._get_advanced_orchestrator_class") as mock:
        # The mock returns a class (callable), which returns an instance
        mock_class = MagicMock()
        mock.return_value = mock_class
        yield mock_class


@pytest.fixture
def mock_handlers():
    return MagicMock(), MagicMock()


def test_create_orchestrator_simple_maps_to_advanced(
    mock_settings, mock_handlers, mock_advanced_cls
):
    """Test that 'simple' mode explicitly maps to AdvancedOrchestrator."""
    search, judge = mock_handlers
    # Pass handlers (they are ignored but shouldn't crash)
    orch = create_orchestrator(search_handler=search, judge_handler=judge, mode="simple")

    # Verify AdvancedOrchestrator was created
    mock_advanced_cls.assert_called_once()
    assert orch == mock_advanced_cls.return_value


def test_create_orchestrator_advanced_explicit(mock_settings, mock_handlers, mock_advanced_cls):
    """Test explicit advanced mode."""
    orch = create_orchestrator(mode="advanced")
    # verify instantiated
    mock_advanced_cls.assert_called_once()
    assert orch == mock_advanced_cls.return_value


def test_create_orchestrator_auto_advanced(mock_settings, mock_advanced_cls):
    """Test auto-detect defaults to Advanced (Unified)."""
    # Even with no keys (handled by factory internally), orchestrator factory returns Advanced
    mock_settings.has_openai_key = False  # Simulate no key

    orch = create_orchestrator()
    mock_advanced_cls.assert_called_once()
    assert orch == mock_advanced_cls.return_value
