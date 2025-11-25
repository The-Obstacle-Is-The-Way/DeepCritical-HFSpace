"""Unit tests for custom exceptions."""

from src.utils.exceptions import (
    ConfigurationError,
    DeepCriticalError,
    JudgeError,
    RateLimitError,
    SearchError,
)


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_search_error_is_deepcritical_error(self):
        assert issubclass(SearchError, DeepCriticalError)

    def test_rate_limit_error_is_search_error(self):
        assert issubclass(RateLimitError, SearchError)

    def test_judge_error_is_deepcritical_error(self):
        assert issubclass(JudgeError, DeepCriticalError)

    def test_configuration_error_is_deepcritical_error(self):
        assert issubclass(ConfigurationError, DeepCriticalError)

    def test_subclass_caught_as_base(self):
        """Verify subclasses can be caught via DeepCriticalError."""
        try:
            raise RateLimitError("rate limited")
        except DeepCriticalError as exc:
            assert isinstance(exc, RateLimitError)
            assert isinstance(exc, DeepCriticalError)
