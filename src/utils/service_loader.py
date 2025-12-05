"""Service loader utility for safe, lazy loading of optional services.

This module handles the import and initialization of services that may
have missing optional dependencies (like Modal or Sentence Transformers),
preventing the application from crashing if they are not available.

Design Patterns:
- Factory Method: get_embedding_service() creates appropriate service
- Strategy Pattern: Selects between EmbeddingService and LlamaIndexRAGService
"""

import threading
from typing import TYPE_CHECKING

import structlog

from src.utils.config import settings

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol

logger = structlog.get_logger()


def warmup_services() -> None:
    """Pre-warm expensive services in a background thread.

    This reduces the "cold start" latency for the first user request by
    loading heavy models (like SentenceTransformer or LlamaIndex) into memory
    during application startup.
    """

    def _warmup() -> None:
        logger.info("🔥 Warmup: Starting background service initialization...")
        try:
            # Trigger model loading (cached globally)
            get_embedding_service_if_available()
            logger.info("🔥 Warmup: Embedding service ready")
        except Exception as e:
            logger.warning("🔥 Warmup: Failed to warm up services", error=str(e))

    # Run in daemon thread so it doesn't block shutdown
    thread = threading.Thread(target=_warmup, daemon=True)
    thread.start()


def get_embedding_service(api_key: str | None = None) -> "EmbeddingServiceProtocol":
    """Get the best available embedding service.

    Strategy selection (ordered by preference):
    1. LlamaIndexRAGService if OPENAI_API_KEY present (better quality + persistence)
    2. EmbeddingService (free, local, in-memory) as fallback

    Design Pattern: Factory Method + Strategy Pattern
    - Factory Method: Creates service instance
    - Strategy Pattern: Selects between implementations at runtime

    Args:
        api_key: Optional BYOK key. If starts with 'sk-', enables Premium tier.

    Returns:
        EmbeddingServiceProtocol: Either LlamaIndexRAGService or EmbeddingService

    Raises:
        ImportError: If no embedding service dependencies are available
    """
    # Determine if we have a valid OpenAI key (BYOK or Env)
    has_openai = False
    if api_key:
        if api_key.startswith("sk-"):
            # OpenAI BYOK
            has_openai = True
    elif settings.has_openai_key:
        has_openai = True

    # Try premium tier first (OpenAI + persistence)
    if has_openai:
        try:
            from src.services.llamaindex_rag import get_rag_service

            # Pass api_key to service (it handles precedence: api_key > env)
            service = get_rag_service(api_key=api_key)
            logger.info(
                "Using LlamaIndex RAG service",
                tier="premium",
                persistence="enabled",
                embeddings="openai",
                byok=bool(api_key),
            )
            return service
        except ImportError as e:
            logger.info(
                "LlamaIndex deps not installed, falling back to local embeddings",
                missing=str(e),
            )
        except Exception as e:
            logger.warning(
                "LlamaIndex service failed to initialize, falling back",
                error=str(e),
                error_type=type(e).__name__,
            )

    # Fallback to free tier (local embeddings, in-memory)
    try:
        from src.services.embeddings import get_embedding_service as get_local_service

        local_service = get_local_service()
        logger.info(
            "Using local embedding service",
            tier="free",
            persistence="disabled",
            embeddings="sentence-transformers",
        )
        return local_service
    except ImportError as e:
        logger.error(
            "No embedding service available",
            error=str(e),
        )
        raise ImportError(
            "No embedding service available. Install either:\n"
            "  - uv sync --extra embeddings (for local embeddings)\n"
            "  - uv sync --extra rag (for LlamaIndex with OpenAI)"
        ) from e


def get_embedding_service_if_available(
    api_key: str | None = None,
) -> "EmbeddingServiceProtocol | None":
    """Safely attempt to load and initialize an embedding service.

    Unlike get_embedding_service(), this function returns None instead of
    raising ImportError when no service is available.

    Args:
        api_key: Optional BYOK key to pass to service factory.

    Returns:
        EmbeddingServiceProtocol instance if dependencies are met, else None.
    """
    try:
        return get_embedding_service(api_key=api_key)
    except ImportError as e:
        logger.info(
            "Embedding service not available (optional dependencies missing)",
            missing_dependency=str(e),
        )
    except Exception as e:
        logger.warning(
            "Embedding service initialization failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
        )
    return None
