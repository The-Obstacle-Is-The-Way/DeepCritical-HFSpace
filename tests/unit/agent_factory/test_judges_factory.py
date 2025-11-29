"""Unit tests for Judge Factory and Model Selection."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit
from pydantic_ai.models.anthropic import AnthropicModel

# We expect this import to exist after we implement it, or we mock it if it's not there yet
# For TDD, we assume we will use the library class
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent_factory.judges import get_model


@pytest.fixture
def mock_settings():
    with patch("src.agent_factory.judges.settings", autospec=True) as mock_settings:
        yield mock_settings


def test_get_model_openai(mock_settings):
    """Test that OpenAI model is returned when provider is openai."""
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = "sk-test"
    mock_settings.openai_model = "gpt-5.1"

    model = get_model()
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-5.1"


def test_get_model_anthropic(mock_settings):
    """Test that Anthropic model is returned when provider is anthropic."""
    mock_settings.llm_provider = "anthropic"
    mock_settings.anthropic_api_key = "sk-ant-test"
    mock_settings.anthropic_model = "claude-sonnet-4-5-20250929"

    model = get_model()
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-4-5-20250929"


def test_get_model_huggingface(mock_settings):
    """Test that HuggingFace model is returned when provider is huggingface."""
    mock_settings.llm_provider = "huggingface"
    mock_settings.hf_token = "hf_test_token"
    mock_settings.huggingface_model = "meta-llama/Llama-3.1-70B-Instruct"

    model = get_model()
    assert isinstance(model, HuggingFaceModel)
    assert model.model_name == "meta-llama/Llama-3.1-70B-Instruct"


def test_get_model_default_fallback(mock_settings):
    """Test fallback to OpenAI if provider is unknown."""
    mock_settings.llm_provider = "unknown_provider"
    mock_settings.openai_api_key = "sk-test"
    mock_settings.openai_model = "gpt-5.1"

    model = get_model()
    assert isinstance(model, OpenAIChatModel)
