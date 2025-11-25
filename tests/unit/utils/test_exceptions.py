"""Unit tests for custom exceptions."""

from src.utils.exceptions import (
    DeepCriticalError,
    JudgeError,
    RateLimitError,
    SearchError,
)


class TestExceptions:
    def test_search_error_is_deepcritical_error(self):
        assert issubclass(SearchError, DeepCriticalError)

    def test_rate_limit_error_is_search_error(self):
        assert issubclass(RateLimitError, SearchError)

    def test_judge_error_is_deepcritical_error(self):
        assert issubclass(JudgeError, DeepCriticalError)
