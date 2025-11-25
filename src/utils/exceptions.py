"""Custom exceptions for DeepCritical."""


class DeepCriticalError(Exception):
    """Base exception for all DeepCritical errors."""

    pass


class SearchError(DeepCriticalError):
    """Raised when a search operation fails."""

    pass


class JudgeError(DeepCriticalError):
    """Raised when the judge fails to assess evidence."""

    pass


class ConfigurationError(DeepCriticalError):
    """Raised when configuration is invalid."""

    pass


class RateLimitError(SearchError):
    """Raised when we hit API rate limits."""

    pass
