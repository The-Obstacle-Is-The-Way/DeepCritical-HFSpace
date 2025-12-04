"""
Client Provider Registry for unified provider selection.

Implements the Strategy Pattern to decouple client creation from the factory.
"""

from typing import Any, ClassVar, Protocol

import structlog
from agent_framework import BaseChatClient

from src.utils.config import Settings

logger = structlog.get_logger()


class ClientProvider(Protocol):
    """Protocol for LLM client providers."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'huggingface')."""
        ...

    def can_handle(
        self, provider_name: str | None, api_key: str | None, settings: Settings
    ) -> bool:
        """Determine if this provider should handle the request."""
        ...

    def create(
        self,
        settings: Settings,
        api_key: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> BaseChatClient:
        """Create the client instance."""
        ...


class ProviderRegistry:
    """Registry for managing available LLM providers."""

    _providers: ClassVar[list[ClientProvider]] = []

    @classmethod
    def register(cls, provider: ClientProvider) -> None:
        """Register a new provider strategy."""
        cls._providers.append(provider)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()

    @classmethod
    def get_client(
        cls,
        settings: Settings,
        provider: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> BaseChatClient:
        """
        Find and execute the appropriate provider strategy.

        Args:
            settings: Application settings
            provider: Explicit provider name
            api_key: Optional API key
            model_id: Optional model ID
            **kwargs: Additional arguments for the client

        Returns:
            Configured BaseChatClient

        Raises:
            ValueError: If no provider can handle the request
        """
        # Normalize provider name
        normalized_provider = provider.lower() if provider else None

        for p in cls._providers:
            if p.can_handle(normalized_provider, api_key, settings):
                logger.info(f"Using {p.name} Chat Client")
                return p.create(settings, api_key, model_id, **kwargs)

        raise ValueError(f"No suitable provider found for provider={provider}")
