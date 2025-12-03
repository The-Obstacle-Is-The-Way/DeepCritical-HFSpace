"""Chat Client Factory for unified provider selection."""

from typing import Any

import structlog
from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatClient

from src.clients.huggingface import HuggingFaceChatClient
from src.utils.config import settings

logger = structlog.get_logger()


def get_chat_client(
    provider: str | None = None,
    api_key: str | None = None,
    model_id: str | None = None,
    **kwargs: Any,
) -> BaseChatClient:
    """
    Factory for creating chat clients.

    Auto-detection priority:
    1. Explicit provider parameter
    2. API key prefix detection (sk- → OpenAI, sk-ant- → Anthropic)
    3. OpenAI key from env (Best Function Calling)
    4. Gemini key from env (Best Context/Cost)
    5. HuggingFace (Free Fallback)

    Args:
        provider: Force specific provider ("openai", "gemini", "huggingface")
        api_key: Override API key for the provider (auto-detects provider from prefix)
        model_id: Override default model ID
        **kwargs: Additional arguments for the client

    Returns:
        Configured BaseChatClient instance (Namespace Neutral)

    Raises:
        ValueError: If an unsupported provider is explicitly requested
        NotImplementedError: If Gemini or Anthropic is requested (not yet implemented)
    """
    # Normalize provider to lowercase for case-insensitive matching
    normalized = provider.lower() if provider is not None else None

    # FIX: Auto-detect provider from API key prefix when not explicitly set
    # This enables BYOK (Bring Your Own Key) from Gradio without explicit provider
    # Order matters: "sk-ant-" must be checked before "sk-" (both start with "sk-")
    if normalized is None and api_key:
        if api_key.startswith("sk-ant-"):
            normalized = "anthropic"
        elif api_key.startswith("sk-"):
            normalized = "openai"
        # HF tokens start with "hf_" - no auto-detection needed (falls through to default)

    # Validate explicit provider requests early
    valid_providers = (None, "openai", "anthropic", "gemini", "huggingface")
    if normalized not in valid_providers:
        raise ValueError(f"Unsupported provider: {provider!r}")

    # 1. OpenAI (Standard / Paid Tier)
    if normalized == "openai" or (normalized is None and settings.has_openai_key):
        logger.info("Using OpenAI Chat Client")
        return OpenAIChatClient(
            model_id=model_id or settings.openai_model,
            api_key=api_key or settings.openai_api_key,
            **kwargs,
        )

    # 2. Anthropic (Detected from sk-ant- prefix or explicit)
    if normalized == "anthropic":
        # Anthropic key was detected or explicitly requested - fail loudly
        raise NotImplementedError(
            "Anthropic client not yet implemented. "
            "Use OpenAI key (sk-...) or leave empty for free HuggingFace tier."
        )

    # 3. Gemini (High Performance / Alternative)
    if normalized == "gemini":
        # Explicit request for Gemini - fail loudly
        raise NotImplementedError("Gemini client not yet implemented (Planned Phase 4)")

    if normalized is None and settings.has_gemini_key:
        # Implicit (has key but not explicit) - log warning and fall through
        logger.warning("Gemini key detected but client not yet implemented; falling back")

    # 4. HuggingFace (Free Fallback)
    # This is the default if no other keys are present
    logger.info("Using HuggingFace Chat Client (Free Tier)")
    return HuggingFaceChatClient(
        model_id=model_id or settings.huggingface_model,
        api_key=api_key or settings.hf_token,
        **kwargs,
    )
