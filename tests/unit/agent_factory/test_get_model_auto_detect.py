import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent_factory.judges import get_model
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError


class TestGetModelAutoDetect:
    """Test that get_model() auto-detects available providers."""

    def test_returns_openai_when_key_present(self, monkeypatch):
        """OpenAI key present → OpenAI model."""
        # Mock the settings properties (settings is a singleton)
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)

        model = get_model()
        assert isinstance(model, OpenAIChatModel)

    def test_returns_anthropic_when_only_anthropic_key(self, monkeypatch):
        """Only Anthropic key → Anthropic model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
        monkeypatch.setattr(settings, "hf_token", None)

        model = get_model()
        assert isinstance(model, AnthropicModel)

    def test_returns_huggingface_when_hf_token_present(self, monkeypatch):
        """HF_TOKEN present (no paid keys) → HuggingFace model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")

        model = get_model()
        assert isinstance(model, HuggingFaceModel)

    def test_raises_error_when_no_keys(self, monkeypatch):
        """No keys at all → ConfigurationError."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)

        with pytest.raises(ConfigurationError) as exc_info:
            get_model()

        assert "No LLM API key configured" in str(exc_info.value)

    def test_openai_takes_priority_over_anthropic(self, monkeypatch):
        """Both keys present → OpenAI wins."""
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")

        model = get_model()
        assert isinstance(model, OpenAIChatModel)
