"""Tests for AdvancedOrchestrator configuration."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.orchestrators.advanced import AdvancedOrchestrator
from src.utils.config import Settings


@pytest.mark.unit
class TestAdvancedOrchestratorConfig:
    """Tests for configuration options."""

    @patch("src.orchestrators.advanced.get_chat_client")
    def test_default_max_rounds_is_five(self, mock_get_client) -> None:
        """Default max_rounds should be 5 from settings."""
        mock_get_client.return_value = MagicMock()
        orch = AdvancedOrchestrator()
        assert orch._max_rounds == 5

    @patch("src.orchestrators.advanced.get_chat_client")
    def test_explicit_max_rounds_overrides_settings(self, mock_get_client) -> None:
        """Explicit parameter should override settings."""
        mock_get_client.return_value = MagicMock()
        orch = AdvancedOrchestrator(max_rounds=7)
        assert orch._max_rounds == 7

    @patch("src.orchestrators.advanced.get_chat_client")
    def test_timeout_default_is_five_minutes(self, mock_get_client) -> None:
        """Default timeout should be 300s (5 min) from settings."""
        mock_get_client.return_value = MagicMock()
        orch = AdvancedOrchestrator()
        assert orch._timeout_seconds == 300.0

    @patch("src.orchestrators.advanced.get_chat_client")
    def test_explicit_timeout_overrides_settings(self, mock_get_client) -> None:
        """Explicit timeout parameter should override settings."""
        mock_get_client.return_value = MagicMock()
        orch = AdvancedOrchestrator(timeout_seconds=120.0)
        assert orch._timeout_seconds == 120.0


@pytest.mark.unit
class TestSettingsValidation:
    """Tests for pydantic Settings validation (fail-fast behavior)."""

    def test_invalid_max_rounds_type_raises(self) -> None:
        """Non-integer ADVANCED_MAX_ROUNDS should fail at startup."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_max_rounds="not_a_number")  # type: ignore[arg-type]
        assert "advanced_max_rounds" in str(exc_info.value)

    def test_zero_max_rounds_raises(self) -> None:
        """ADVANCED_MAX_ROUNDS=0 should fail validation (ge=1)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_max_rounds=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_negative_max_rounds_raises(self) -> None:
        """Negative ADVANCED_MAX_ROUNDS should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_max_rounds=-5)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_rounds_above_limit_raises(self) -> None:
        """ADVANCED_MAX_ROUNDS > 20 should fail validation (le=20)."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_max_rounds=100)
        assert "less than or equal to 20" in str(exc_info.value)

    def test_valid_max_rounds_accepted(self) -> None:
        """Valid ADVANCED_MAX_ROUNDS should be accepted."""
        s = Settings(advanced_max_rounds=10)
        assert s.advanced_max_rounds == 10

    def test_timeout_too_low_raises(self) -> None:
        """ADVANCED_TIMEOUT < 60 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_timeout=30.0)
        assert "greater than or equal to 60" in str(exc_info.value)

    def test_timeout_too_high_raises(self) -> None:
        """ADVANCED_TIMEOUT > 900 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(advanced_timeout=1000.0)
        assert "less than or equal to 900" in str(exc_info.value)
