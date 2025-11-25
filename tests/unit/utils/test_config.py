"""Unit tests for configuration loading."""
import pytest
from unittest.mock import patch
import os


class TestSettings:
    """Tests for Settings class."""

    def test_default_max_iterations(self):
        """Settings should have default max_iterations of 10."""
        from src.utils.config import Settings

        # Clear any env vars
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.max_iterations == 10

    def test_max_iterations_from_env(self):
        """Settings should read MAX_ITERATIONS from env."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {"MAX_ITERATIONS": "25"}):
            settings = Settings()
            assert settings.max_iterations == 25

    def test_invalid_max_iterations_raises(self):
        """Settings should reject invalid max_iterations."""
        from src.utils.config import Settings
        from pydantic import ValidationError

        with patch.dict(os.environ, {"MAX_ITERATIONS": "100"}):
            with pytest.raises(ValidationError):
                Settings()  # 100 > 50 (max)

    def test_get_api_key_openai(self):
        """get_api_key should return OpenAI key when provider is openai."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key"
        }):
            settings = Settings()
            assert settings.get_api_key() == "sk-test-key"

    def test_get_api_key_missing_raises(self):
        """get_api_key should raise when key is not set."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                settings.get_api_key()
