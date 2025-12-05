import pytest
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent_factory.judges import get_model
from src.utils.config import settings


class TestGetModelAutoDetect:
    """Test that get_model() auto-detects available providers."""

    def test_returns_openai_when_key_present(self, monkeypatch):
        """OpenAI key present → OpenAI model."""
        # Mock the settings properties (settings is a singleton)
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "hf_token", None)

        model = get_model()
        assert isinstance(model, OpenAIChatModel)

    def test_byok_openai_key_returns_openai_model(self, monkeypatch):
        """BYOK: api_key='sk-...' → OpenAI model (regardless of env vars)."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)

        model = get_model(api_key="sk-byok-test-key")
        assert isinstance(model, OpenAIChatModel)

    def test_returns_huggingface_when_hf_token_present(self, monkeypatch):
        """HF_TOKEN present (no paid keys) → HuggingFace model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")

        model = get_model()
        assert isinstance(model, HuggingFaceModel)

    def test_raises_when_no_api_keys_available(self, monkeypatch):
        """No keys at all → RuntimeError with helpful message."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)
        monkeypatch.setattr(settings, "huggingface_model", "Qwen/Qwen2.5-7B-Instruct")
        # Also ensure HF_TOKEN env var is not set
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Should raise clear error when no tokens available

        with pytest.raises(RuntimeError) as exc_info:
            get_model()
        assert "No LLM API key available" in str(exc_info.value)
        assert "HF_TOKEN" in str(exc_info.value)

    def test_openai_env_takes_priority_over_huggingface(self, monkeypatch):
        """OpenAI env key present → OpenAI wins over HuggingFace."""
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")

        model = get_model()
        assert isinstance(model, OpenAIChatModel)
