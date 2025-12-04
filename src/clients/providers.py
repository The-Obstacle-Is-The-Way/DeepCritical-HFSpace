"""
LLM Client Provider Strategies.
"""

from typing import Any

from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatClient

from src.clients.huggingface import HuggingFaceChatClient
from src.clients.registry import ClientProvider
from src.utils.config import Settings


class OpenAIProvider(ClientProvider):
    """Strategy for OpenAI client creation."""

    @property
    def name(self) -> str:
        return "OpenAI"

    def can_handle(
        self, provider_name: str | None, api_key: str | None, settings: Settings
    ) -> bool:
        # 1. Explicit request
        if provider_name == "openai":
            return True

        # 2. BYOK Detection (sk-...)
        if provider_name is None and api_key and api_key.startswith("sk-"):
            return True

        # 3. Env Fallback (if no explicit provider)
        if provider_name is None and settings.has_openai_key:
            return True

        return False

    def create(
        self,
        settings: Settings,
        api_key: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> BaseChatClient:
        return OpenAIChatClient(
            model_id=model_id or settings.openai_model,
            api_key=api_key or settings.openai_api_key,
            **kwargs,
        )


class HuggingFaceProvider(ClientProvider):
    """Strategy for HuggingFace client creation (Free Tier Fallback)."""

    @property
    def name(self) -> str:
        return "HuggingFace"

    def can_handle(
        self, provider_name: str | None, api_key: str | None, settings: Settings
    ) -> bool:
        # 1. Explicit request
        if provider_name == "huggingface":
            return True

        # 2. Fallback (Default) - only if NO specific provider requested
        if provider_name is None:
            return True

        return False

    def create(
        self,
        settings: Settings,
        api_key: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> BaseChatClient:
        return HuggingFaceChatClient(
            model_id=model_id or settings.huggingface_model,
            api_key=api_key or settings.hf_token,
            **kwargs,
        )
