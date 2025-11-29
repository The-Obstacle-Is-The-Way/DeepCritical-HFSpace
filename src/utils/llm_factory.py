"""Centralized LLM client factory.

This module provides factory functions for creating LLM clients,
ensuring consistent configuration and clear error messages.

Why Magentic requires OpenAI:
- Magentic agents use the @ai_function decorator for tool calling
- This requires structured function calling protocol (tools, tool_choice)
- OpenAI's API supports this natively
- Anthropic/HuggingFace Inference APIs are text-in/text-out only
"""

from typing import TYPE_CHECKING, Any

from src.utils.config import settings
from src.utils.exceptions import ConfigurationError

if TYPE_CHECKING:
    from agent_framework.openai import OpenAIChatClient


def get_magentic_client() -> "OpenAIChatClient":
    """
    Get the OpenAI client for Magentic agents.

    Magentic requires OpenAI because it uses function calling protocol:
    - @ai_function decorators define callable tools
    - LLM returns structured tool calls (not just text)
    - Requires OpenAI's tools/function_call API support

    Raises:
        ConfigurationError: If OPENAI_API_KEY is not set

    Returns:
        Configured OpenAIChatClient for Magentic agents
    """
    # Import here to avoid requiring agent-framework for simple mode
    from agent_framework.openai import OpenAIChatClient

    api_key = settings.get_openai_api_key()

    return OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=api_key,
    )


def get_pydantic_ai_model() -> Any:
    """
    Get the appropriate model for pydantic-ai based on configuration.

    Uses the configured LLM_PROVIDER to select between OpenAI and Anthropic.
    This is used by simple mode components (JudgeHandler, etc.)

    Returns:
        Configured pydantic-ai model
    """
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY not set for pydantic-ai")
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        return OpenAIChatModel(settings.openai_model, provider=provider)

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY not set for pydantic-ai")
        anthropic_provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model, provider=anthropic_provider)

    raise ConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")


def check_magentic_requirements() -> None:
    """
    Check if Magentic mode requirements are met.

    Raises:
        ConfigurationError: If requirements not met
    """
    if not settings.has_openai_key:
        raise ConfigurationError(
            "Magentic mode requires OPENAI_API_KEY for function calling support. "
            "Anthropic and HuggingFace Inference do not support the structured "
            "function calling protocol that Magentic agents require. "
            "Use mode='simple' for other LLM providers."
        )


def check_simple_mode_requirements() -> None:
    """
    Check if simple mode requirements are met.

    Simple mode supports both OpenAI and Anthropic.

    Raises:
        ConfigurationError: If no LLM API key is configured
    """
    if not settings.has_any_llm_key:
        raise ConfigurationError(
            "No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
