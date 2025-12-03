"""Unit tests for ChatClientFactory (SPEC-16: Unified Architecture)."""

from unittest.mock import MagicMock, patch

import pytest

# Skip if agent-framework-core not installed
pytest.importorskip("agent_framework")


@pytest.mark.unit
class TestChatClientFactory:
    """Test get_chat_client() factory function."""

    def test_returns_openai_client_when_openai_key_available(self) -> None:
        """When OpenAI key is available, should return OpenAIChatClient."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = True
            mock_settings.has_gemini_key = False
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.openai_model = "gpt-5"

            from src.clients.factory import get_chat_client

            client = get_chat_client()

            # Should be OpenAIChatClient
            assert "OpenAI" in type(client).__name__

    def test_returns_huggingface_client_when_no_key_available(self) -> None:
        """When no API key is available, should return HuggingFaceChatClient (free tier)."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from src.clients.factory import get_chat_client

            client = get_chat_client()

            # Should be HuggingFaceChatClient
            assert "HuggingFace" in type(client).__name__

    def test_explicit_provider_openai_overrides_auto_detection(self) -> None:
        """Explicit provider='openai' should use OpenAI even if no env key."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-5"

            from src.clients.factory import get_chat_client

            # Explicit provider with api_key parameter
            client = get_chat_client(provider="openai", api_key="sk-explicit-key")

            assert "OpenAI" in type(client).__name__

    def test_explicit_provider_huggingface(self) -> None:
        """Explicit provider='huggingface' should use HuggingFace."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = True  # Even with OpenAI key available
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from src.clients.factory import get_chat_client

            # Explicit provider forces HuggingFace
            client = get_chat_client(provider="huggingface")

            assert "HuggingFace" in type(client).__name__

    def test_gemini_provider_raises_not_implemented(self) -> None:
        """Explicit provider='gemini' should raise NotImplementedError (Phase 4)."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False

            from src.clients.factory import get_chat_client

            with pytest.raises(NotImplementedError, match="Gemini client not yet implemented"):
                get_chat_client(provider="gemini")

    def test_unsupported_provider_raises_value_error(self) -> None:
        """Unsupported provider should raise ValueError, not silently fallback."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False

            from src.clients.factory import get_chat_client

            with pytest.raises(ValueError, match="Unsupported provider"):
                get_chat_client(provider="invalid_provider")

    def test_anthropic_provider_raises_not_implemented(self) -> None:
        """Anthropic provider should raise NotImplementedError (not yet implemented)."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False

            from src.clients.factory import get_chat_client

            with pytest.raises(NotImplementedError, match="Anthropic client not yet implemented"):
                get_chat_client(provider="anthropic")

    def test_byok_auto_detects_openai_from_key_prefix(self) -> None:
        """BYOK: api_key starting with 'sk-' should auto-select OpenAI without explicit provider.

        This is the critical BYOK (Bring Your Own Key) test case:
        - User enters 'sk-...' key in Gradio
        - No explicit provider parameter
        - No OPENAI_API_KEY in env (settings.has_openai_key = False)
        - Should auto-detect OpenAI from the key prefix
        """
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False  # No env key
            mock_settings.has_gemini_key = False
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-5"

            from src.clients.factory import get_chat_client

            # BYOK: Pass api_key without explicit provider
            client = get_chat_client(api_key="sk-user-provided-key")

            # Should auto-detect OpenAI from 'sk-' prefix
            assert "OpenAI" in type(client).__name__

    def test_byok_auto_detects_anthropic_from_key_prefix(self) -> None:
        """BYOK: api_key starting with 'sk-ant-' should auto-detect Anthropic.

        Anthropic keys start with 'sk-ant-' which is a superset of 'sk-'.
        Detection must check 'sk-ant-' first to avoid misdetecting as OpenAI.
        """
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False

            from src.clients.factory import get_chat_client

            # BYOK: Anthropic key should raise NotImplementedError (not fall to HuggingFace!)
            with pytest.raises(NotImplementedError, match="Anthropic client not yet implemented"):
                get_chat_client(api_key="sk-ant-user-anthropic-key")

    def test_byok_hf_token_falls_through_to_huggingface(self) -> None:
        """BYOK: HuggingFace tokens (hf_...) should use HuggingFace client."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False
            mock_settings.huggingface_model = "Qwen/Qwen2.5-7B-Instruct"
            mock_settings.hf_token = None

            from src.clients.factory import get_chat_client

            # HF tokens don't trigger auto-detection, falls through to HuggingFace
            client = get_chat_client(api_key="hf_user_provided_token")

            assert "HuggingFace" in type(client).__name__

    def test_provider_is_case_insensitive(self) -> None:
        """Provider matching should be case-insensitive."""
        with patch("src.clients.factory.settings") as mock_settings:
            mock_settings.has_openai_key = False
            mock_settings.has_gemini_key = False
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-5"

            from src.clients.factory import get_chat_client

            # "OpenAI" should work same as "openai"
            client = get_chat_client(provider="OpenAI", api_key="sk-test")
            assert "OpenAI" in type(client).__name__

            # "HUGGINGFACE" should work same as "huggingface"
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None
            client = get_chat_client(provider="HUGGINGFACE")
            assert "HuggingFace" in type(client).__name__


@pytest.mark.unit
class TestHuggingFaceChatClient:
    """Test HuggingFaceChatClient adapter."""

    def test_initialization_with_defaults(self) -> None:
        """Should initialize with default model from settings."""
        with patch("src.clients.huggingface.settings") as mock_settings:
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from src.clients.huggingface import HuggingFaceChatClient

            client = HuggingFaceChatClient()

            assert client.model_id == "meta-llama/Llama-3.1-70B-Instruct"

    def test_initialization_with_custom_model(self) -> None:
        """Should accept custom model_id."""
        with patch("src.clients.huggingface.settings") as mock_settings:
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from src.clients.huggingface import HuggingFaceChatClient

            client = HuggingFaceChatClient(model_id="mistralai/Mistral-7B-Instruct-v0.3")

            assert client.model_id == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_convert_messages_basic(self) -> None:
        """Should convert ChatMessage list to HuggingFace format."""
        with patch("src.clients.huggingface.settings") as mock_settings:
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from agent_framework import ChatMessage

            from src.clients.huggingface import HuggingFaceChatClient

            client = HuggingFaceChatClient()

            # Create mock messages (include contents=None for tool call processing)
            messages = [
                MagicMock(spec=ChatMessage, role="user", text="Hello", contents=None),
                MagicMock(spec=ChatMessage, role="assistant", text="Hi there!", contents=None),
            ]

            result = client._convert_messages(messages)

            assert len(result) == 2
            assert result[0] == {"role": "user", "content": "Hello"}
            assert result[1] == {"role": "assistant", "content": "Hi there!"}

    def test_convert_messages_handles_role_enum(self) -> None:
        """Should extract .value from Role enum, not stringify the enum itself."""
        with patch("src.clients.huggingface.settings") as mock_settings:
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from enum import Enum

            from agent_framework import ChatMessage

            from src.clients.huggingface import HuggingFaceChatClient

            # Simulate a Role enum like agent_framework might use
            class Role(Enum):
                USER = "user"
                ASSISTANT = "assistant"

            client = HuggingFaceChatClient()

            # Create mock message with enum role
            mock_msg = MagicMock(spec=ChatMessage)
            mock_msg.role = Role.USER  # Enum, not string
            mock_msg.text = "Hello"
            mock_msg.contents = None  # Required for tool call processing

            result = client._convert_messages([mock_msg])

            # Should be "user", NOT "Role.USER"
            assert result[0]["role"] == "user"
            assert "Role" not in result[0]["role"]

    def test_inherits_from_base_chat_client(self) -> None:
        """Should inherit from agent_framework.BaseChatClient."""
        with patch("src.clients.huggingface.settings") as mock_settings:
            mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.hf_token = None

            from agent_framework import BaseChatClient

            from src.clients.huggingface import HuggingFaceChatClient

            client = HuggingFaceChatClient()

            assert isinstance(client, BaseChatClient)
