"""Custom exceptions for DeepBoner."""


class DeepBonerError(Exception):
    """Base exception for all DeepBoner errors."""

    pass


class SearchError(DeepBonerError):
    """Raised when a search operation fails."""

    pass


class JudgeError(DeepBonerError):
    """Raised when the judge fails to assess evidence."""

    pass


class ConfigurationError(DeepBonerError):
    """Raised when configuration is invalid."""

    pass


class RateLimitError(SearchError):
    """Raised when we hit API rate limits."""

    pass


# Backwards compatibility alias
DeepCriticalError = DeepBonerError
