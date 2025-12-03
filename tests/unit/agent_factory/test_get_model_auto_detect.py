import pytest
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent_factory.judges import get_model
from src.utils.config import settings


class TestGetModelAutoDetect:
    """Test that get_model() auto-detects available providers.

    NOTE: Anthropic is NOT supported (no embeddings API).
    See P3_REMOVE_ANTHROPIC_PARTIAL_WIRING.md.
    """

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

    def test_byok_anthropic_key_raises_not_implemented(self, monkeypatch):
        """BYOK: api_key='sk-ant-...' → NotImplementedError (Anthropic not supported)."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", None)

        with pytest.raises(NotImplementedError) as exc_info:
            get_model(api_key="sk-ant-test-key")

        assert "Anthropic is not supported" in str(exc_info.value)

    def test_returns_huggingface_when_hf_token_present(self, monkeypatch):
        """HF_TOKEN present (no paid keys) → HuggingFace model."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")

        model = get_model()
        assert isinstance(model, HuggingFaceModel)

    def test_falls_through_to_huggingface_when_no_keys(self, monkeypatch):
        """No keys at all → Falls through to HuggingFace (free tier)."""
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")
        monkeypatch.setattr(settings, "huggingface_model", "Qwen/Qwen2.5-7B-Instruct")

        # Should return HuggingFace model (free tier)
        model = get_model()
        assert isinstance(model, HuggingFaceModel)

    def test_openai_env_takes_priority_over_huggingface(self, monkeypatch):
        """OpenAI env key present → OpenAI wins over HuggingFace."""
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(settings, "hf_token", "hf_test_token")

        model = get_model()
        assert isinstance(model, OpenAIChatModel)
