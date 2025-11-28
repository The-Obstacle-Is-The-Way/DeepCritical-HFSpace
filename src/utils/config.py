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
    llm_provider: Literal["openai", "anthropic", "huggingface"] = Field(
        default="openai", description="Which LLM provider to use"
    )
    openai_model: str = Field(default="gpt-5.1", description="OpenAI model name")
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
    embedding_provider: Literal["openai", "local", "huggingface"] = Field(
        default="local",
        description="Embedding provider to use",
    )
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model ID",
    )

    # HuggingFace Configuration
    huggingface_api_key: str | None = Field(
        default=None, description="HuggingFace API token (HF_TOKEN or HUGGINGFACE_API_KEY)"
    )
    huggingface_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Default HuggingFace model ID for inference",
    )

    # PubMed Configuration
    ncbi_api_key: str | None = Field(
        default=None, description="NCBI API key for higher rate limits"
    )

    # Web Search Configuration
    web_search_provider: Literal["serper", "searchxng", "brave", "tavily", "duckduckgo"] = Field(
        default="duckduckgo",
        description="Web search provider to use",
    )
    serper_api_key: str | None = Field(default=None, description="Serper API key for Google search")
    searchxng_host: str | None = Field(default=None, description="SearchXNG host URL")
    brave_api_key: str | None = Field(default=None, description="Brave Search API key")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key")

    # Agent Configuration
    max_iterations: int = Field(default=10, ge=1, le=50)
    search_timeout: int = Field(default=30, description="Seconds to wait for search")
    use_graph_execution: bool = Field(
        default=False, description="Use graph-based execution for research flows"
    )

    # Budget & Rate Limiting Configuration
    default_token_limit: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        description="Default token budget per research loop",
    )
    default_time_limit_minutes: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Default time limit per research loop (minutes)",
    )
    default_iterations_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default iterations limit per research loop",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # External Services
    modal_token_id: str | None = Field(default=None, description="Modal token ID")
    modal_token_secret: str | None = Field(default=None, description="Modal token secret")
    chroma_db_path: str = Field(default="./chroma_db", description="ChromaDB storage path")
    chroma_db_persist: bool = Field(
        default=True,
        description="Whether to persist ChromaDB to disk",
    )
    chroma_db_host: str | None = Field(
        default=None,
        description="ChromaDB server host (for remote ChromaDB)",
    )
    chroma_db_port: int | None = Field(
        default=None,
        description="ChromaDB server port (for remote ChromaDB)",
    )

    # RAG Service Configuration
    rag_collection_name: str = Field(
        default="deepcritical_evidence",
        description="ChromaDB collection name for RAG",
    )
    rag_similarity_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to retrieve from RAG",
    )
    rag_auto_ingest: bool = Field(
        default=True,
        description="Automatically ingest evidence into RAG",
    )

    @property
    def modal_available(self) -> bool:
        """Check if Modal credentials are configured."""
        return bool(self.modal_token_id and self.modal_token_secret)

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
    def has_huggingface_key(self) -> bool:
        """Check if HuggingFace token is available."""
        return bool(self.hf_token)

    @property
    def has_any_llm_key(self) -> bool:
        """Check if any LLM API key is available."""
        return self.has_openai_key or self.has_anthropic_key or self.has_huggingface_key

    @property
    def has_huggingface_key(self) -> bool:
        """Check if HuggingFace API key is available."""
        return bool(self.huggingface_api_key)

    @property
    def web_search_available(self) -> bool:
        """Check if web search is available (either no-key provider or API key present)."""
        if self.web_search_provider == "duckduckgo":
            return True  # No API key required
        if self.web_search_provider == "serper":
            return bool(self.serper_api_key)
        if self.web_search_provider == "searchxng":
            return bool(self.searchxng_host)
        if self.web_search_provider == "brave":
            return bool(self.brave_api_key)
        if self.web_search_provider == "tavily":
            return bool(self.tavily_api_key)
        return False


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
