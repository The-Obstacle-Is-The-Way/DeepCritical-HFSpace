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


class SynthesisError(DeepBonerError):
    """Raised when report synthesis fails after trying all available models.

    Attributes:
        message: Human-readable error description
        attempted_models: List of model IDs that were tried
        errors: List of error messages from each failed attempt
    """

    def __init__(
        self,
        message: str,
        attempted_models: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Initialize SynthesisError with context.

        Args:
            message: Human-readable error description
            attempted_models: Models that were tried before failing
            errors: Error messages from each failed model attempt
        """
        super().__init__(message)
        self.attempted_models = attempted_models or []
        self.errors = errors or []
