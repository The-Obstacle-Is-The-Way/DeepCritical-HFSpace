"""Chat Client Factory for unified provider selection."""

from typing import Any

import structlog
from agent_framework import BaseChatClient

from src.clients.providers import HuggingFaceProvider, OpenAIProvider
from src.clients.registry import ProviderRegistry
from src.utils.config import settings

logger = structlog.get_logger()

# Register strategies in order of priority
# 1. OpenAI (Specific key or Env)
ProviderRegistry.register(OpenAIProvider())
# 2. HuggingFace (Free Fallback)
ProviderRegistry.register(HuggingFaceProvider())


def get_chat_client(
    provider: str | None = None,
    api_key: str | None = None,
    model_id: str | None = None,
    **kwargs: Any,
) -> BaseChatClient:
    """
    Factory for creating chat clients.

    Delegates to ProviderRegistry for strategy selection.

    Auto-detection priority (via Registry):
    1. Explicit provider parameter
    2. API key prefix detection (sk- → OpenAI)
    3. OpenAI key from env
    4. HuggingFace (Free Fallback)

    Args:
        provider: Force specific provider ("openai", "huggingface")
        api_key: Override API key
        model_id: Override default model ID
        **kwargs: Additional arguments for the client

    Returns:
        Configured BaseChatClient instance

    Raises:
        ValueError: If an unsupported provider is requested
    """
    return ProviderRegistry.get_client(
        settings=settings,
        provider=provider,
        api_key=api_key,
        model_id=model_id,
        **kwargs,
    )
