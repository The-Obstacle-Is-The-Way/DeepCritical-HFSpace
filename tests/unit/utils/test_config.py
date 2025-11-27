"""Unit tests for configuration loading."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.utils.config import Settings
from src.utils.exceptions import ConfigurationError


class TestSettings:
    """Tests for Settings class."""

    def test_default_max_iterations(self):
        """Settings should have default max_iterations of 10."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.max_iterations == 10  # noqa: PLR2004

    def test_max_iterations_from_env(self):
        """Settings should read MAX_ITERATIONS from env."""
        with patch.dict(os.environ, {"MAX_ITERATIONS": "25"}):
            settings = Settings()
            assert settings.max_iterations == 25  # noqa: PLR2004

    def test_invalid_max_iterations_raises(self):
        """Settings should reject invalid max_iterations."""
        with patch.dict(os.environ, {"MAX_ITERATIONS": "100"}):
            with pytest.raises(ValidationError):
                Settings()  # 100 > 50 (max)

    def test_get_api_key_openai(self):
        """get_api_key should return OpenAI key when provider is openai."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-key"}):
            settings = Settings()
            assert settings.get_api_key() == "sk-test-key"

    def test_get_api_key_openai_missing_raises(self):
        """get_api_key should raise ConfigurationError when OpenAI key is not set."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not set"):
                settings.get_api_key()

    def test_get_api_key_anthropic(self):
        """get_api_key should return Anthropic key when provider is anthropic."""
        with patch.dict(
            os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant-test-key"}
        ):
            settings = Settings()
            assert settings.get_api_key() == "sk-ant-test-key"

    def test_get_api_key_anthropic_missing_raises(self):
        """get_api_key should raise ConfigurationError when Anthropic key is not set."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            settings = Settings()
            with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY not set"):
                settings.get_api_key()
