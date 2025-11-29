"""Unit tests for custom exceptions."""

import pytest

pytestmark = pytest.mark.unit

from src.utils.exceptions import (
    ConfigurationError,
    DeepBonerError,
    JudgeError,
    RateLimitError,
    SearchError,
)


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_search_error_is_deepboner_error(self):
        assert issubclass(SearchError, DeepBonerError)

    def test_rate_limit_error_is_search_error(self):
        assert issubclass(RateLimitError, SearchError)

    def test_judge_error_is_deepboner_error(self):
        assert issubclass(JudgeError, DeepBonerError)

    def test_configuration_error_is_deepboner_error(self):
        assert issubclass(ConfigurationError, DeepBonerError)

    def test_subclass_caught_as_base(self):
        """Verify subclasses can be caught via DeepBonerError."""
        try:
            raise RateLimitError("rate limited")
        except DeepBonerError as exc:
            assert isinstance(exc, RateLimitError)
            assert isinstance(exc, DeepBonerError)
