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


class EmbeddingError(DeepBonerError):
    """Raised when embedding or vector store operations fail."""

    pass


class LLMError(DeepBonerError):
    """Raised when LLM operations fail (API errors, parsing errors, etc.)."""

    pass


class QuotaExceededError(LLMError):
    """Raised when LLM API quota is exceeded (402 errors)."""

    pass


class ModalError(DeepBonerError):
    """Raised when Modal sandbox operations fail."""

    pass


class SynthesisError(DeepBonerError):
    """Raised when report synthesis fails."""

    pass
