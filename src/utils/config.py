"""Application configuration using Pydantic Settings."""

import logging
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.domain import ResearchDomain
from src.utils.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Domain configuration
    research_domain: ResearchDomain = ResearchDomain.SEXUAL_HEALTH

    # LLM Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    llm_provider: Literal["openai", "anthropic", "huggingface", "gemini"] = Field(
        default="openai", description="Which LLM provider to use"
    )
    openai_model: str = Field(default="gpt-5", description="OpenAI model name")
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Anthropic model"
    )
    # HuggingFace (free tier)
    huggingface_model: str | None = Field(
        default="meta-llama/Llama-3.1-70B-Instruct", description="HuggingFace model name"
    )
    hf_token: str | None = Field(
        default=None, alias="HF_TOKEN", description="HuggingFace API token"
    )

    # Embedding Configuration
    # Note: OpenAI embeddings require OPENAI_API_KEY (Anthropic has no embeddings API)
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (used by LlamaIndex RAG)",
    )
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model (used by EmbeddingService)",
    )

    # PubMed Configuration
    ncbi_api_key: str | None = Field(
        default=None, description="NCBI API key for higher rate limits"
    )

    # Agent Configuration
    max_iterations: int = Field(default=10, ge=1, le=50)
    advanced_max_rounds: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max coordination rounds for Advanced mode (default 5 for faster demos)",
    )
    advanced_timeout: float = Field(
        default=300.0,
        ge=60.0,
        le=900.0,
        description="Timeout for Advanced mode in seconds (default 5 min)",
    )
    search_timeout: int = Field(default=30, description="Seconds to wait for search")
    magentic_timeout: int = Field(
        default=600,
        description="Timeout for Magentic mode in seconds (deprecated, use advanced_timeout)",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # External Services
    modal_token_id: str | None = Field(default=None, description="Modal token ID")
    modal_token_secret: str | None = Field(default=None, description="Modal token secret")
    chroma_db_path: str = Field(default="./chroma_db", description="ChromaDB storage path")

    @property
    def modal_available(self) -> bool:
        """Check if Modal credentials are configured."""
        return bool(self.modal_token_id and self.modal_token_secret)

    def get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        # Normalize provider for case-insensitive matching
        provider_lower = self.llm_provider.lower() if self.llm_provider else ""

        if provider_lower == "openai":
            if not self.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY not set")
            return self.openai_api_key

        if provider_lower == "anthropic":
            if not self.anthropic_api_key:
                raise ConfigurationError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key

        raise ConfigurationError(f"Unknown LLM provider: {self.llm_provider}")

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key (required for Magentic function calling)."""
        if not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY not set. Magentic mode requires OpenAI for function calling. "
                "Use mode='simple' for other providers."
            )
        return self.openai_api_key

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key)

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.anthropic_api_key)

    @property
    def has_gemini_key(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(self.gemini_api_key)

    @property
    def has_huggingface_key(self) -> bool:
        """Check if HuggingFace token is available."""
        return bool(self.hf_token)

    @property
    def has_any_llm_key(self) -> bool:
        """Check if any LLM API key is available."""
        return (
            self.has_openai_key
            or self.has_anthropic_key
            or self.has_huggingface_key
            or self.has_gemini_key
        )


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
