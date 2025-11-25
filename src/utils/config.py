"""Application configuration using Pydantic Settings."""

import logging
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai", description="Which LLM provider to use"
    )
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    anthropic_model: str = Field(default="claude-sonnet-4-5", description="Anthropic model")

    # PubMed Configuration
    ncbi_api_key: str | None = Field(
        default=None, description="NCBI API key for higher rate limits"
    )

    # Agent Configuration
    max_iterations: int = Field(default=10, ge=1, le=50)
    search_timeout: int = Field(default=30, description="Seconds to wait for search")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY not set")
            return self.openai_api_key

        if self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ConfigurationError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key

        raise ConfigurationError(f"Unknown LLM provider: {self.llm_provider}")


def get_settings() -> Settings:
    """Factory function to get settings (allows mocking in tests)."""
    return Settings()


def configure_logging(settings: Settings) -> None:
    """Configure structured logging with the configured log level."""
    # Set stdlib logging level from settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# Singleton for easy import
settings = get_settings()
