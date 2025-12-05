"""Unit tests for Judge Factory and Model Selection."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent_factory.judges import get_model


@pytest.fixture
def mock_settings():
    with patch("src.agent_factory.judges.settings", autospec=True) as mock_settings:
        yield mock_settings


def test_get_model_openai(mock_settings):
    """Test that OpenAI model is returned when provider is openai."""
    mock_settings.has_openai_key = True
    mock_settings.openai_api_key = "sk-test"
    mock_settings.openai_model = "gpt-5"

    model = get_model()
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-5"


def test_get_model_byok_openai(mock_settings):
    """Test that BYOK OpenAI key returns OpenAI model."""
    mock_settings.has_openai_key = False
    mock_settings.openai_model = "gpt-5"

    # BYOK takes priority over env vars
    model = get_model(api_key="sk-byok-test")
    assert isinstance(model, OpenAIChatModel)


def test_get_model_huggingface(mock_settings):
    """Test that HuggingFace model is returned when no paid keys."""
    mock_settings.has_openai_key = False
    mock_settings.hf_token = "hf_test_token"
    mock_settings.huggingface_model = "Qwen/Qwen2.5-7B-Instruct"

    model = get_model()
    assert isinstance(model, HuggingFaceModel)
    assert model.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_get_model_openai_priority(mock_settings):
    """Test OpenAI takes priority when both keys present."""
    mock_settings.has_openai_key = True
    mock_settings.openai_api_key = "sk-test"
    mock_settings.openai_model = "gpt-5"
    mock_settings.hf_token = "hf_test_token"

    model = get_model()
    assert isinstance(model, OpenAIChatModel)
