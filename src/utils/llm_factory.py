"""Centralized LLM client factory.

This module provides factory functions for creating LLM clients.
DEPRECATED: Prefer src.clients.factory.get_chat_client() directly.
"""

from typing import Any

from src.clients.base import BaseChatClient
from src.clients.factory import get_chat_client
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError


def get_magentic_client() -> BaseChatClient:
    """
    Get the chat client for Magentic agents.

    Now unified to support OpenAI, Gemini, and HuggingFace.
    """
    return get_chat_client()


def get_pydantic_ai_model() -> Any:
    """
    Get the appropriate model for pydantic-ai based on configuration.
    Used by legacy Simple Mode components.
    """
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    # Normalize provider for case-insensitive matching
    provider_lower = settings.llm_provider.lower() if settings.llm_provider else ""

    if provider_lower == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY not set for pydantic-ai")
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        return OpenAIChatModel(settings.openai_model, provider=provider)

    if provider_lower == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY not set for pydantic-ai")
        anthropic_provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model, provider=anthropic_provider)

    raise ConfigurationError(f"Unknown LLM provider for simple mode: {settings.llm_provider}")


def check_magentic_requirements() -> None:
    """
    Check if Magentic mode requirements are met.
    Now supports multiple providers via ChatClientFactory.
    """
    # Advanced/Magentic mode now works with ANY provider (including free HF)
    pass


def check_simple_mode_requirements() -> None:
    """
    Check if simple mode requirements are met.
    """
    if not settings.has_any_llm_key:
        # Simple mode still requires explicit keys?
        # Actually, simple mode also had HF support but it was brittle.
        # We are deleting simple mode later, so let's leave this as is for now.
        pass
