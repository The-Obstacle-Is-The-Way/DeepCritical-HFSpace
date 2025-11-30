"""Service loader utility for safe, lazy loading of optional services.

This module handles the import and initialization of services that may
have missing optional dependencies (like Modal or Sentence Transformers),
preventing the application from crashing if they are not available.

Design Patterns:
- Factory Method: get_embedding_service() creates appropriate service
- Strategy Pattern: Selects between EmbeddingService and LlamaIndexRAGService
"""

from typing import TYPE_CHECKING

import structlog

from src.utils.config import settings

if TYPE_CHECKING:
    from src.services.embedding_protocol import EmbeddingServiceProtocol
    from src.services.statistical_analyzer import StatisticalAnalyzer

logger = structlog.get_logger()


def get_embedding_service() -> "EmbeddingServiceProtocol":
    """Get the best available embedding service.

    Strategy selection (ordered by preference):
    1. LlamaIndexRAGService if OPENAI_API_KEY present (better quality + persistence)
    2. EmbeddingService (free, local, in-memory) as fallback

    Design Pattern: Factory Method + Strategy Pattern
    - Factory Method: Creates service instance
    - Strategy Pattern: Selects between implementations at runtime

    Returns:
        EmbeddingServiceProtocol: Either LlamaIndexRAGService or EmbeddingService

    Raises:
        ImportError: If no embedding service dependencies are available

    Example:
        ```python
        service = get_embedding_service()
        await service.add_evidence("id", "content", {"source": "pubmed"})
        results = await service.search_similar("query", n_results=5)
        unique = await service.deduplicate(evidence_list)
        ```
    """
    # Try premium tier first (OpenAI + persistence)
    if settings.has_openai_key:
        try:
            from src.services.llamaindex_rag import get_rag_service

            service = get_rag_service()
            logger.info(
                "Using LlamaIndex RAG service",
                tier="premium",
                persistence="enabled",
                embeddings="openai",
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
            "  - uv sync --extra modal (for LlamaIndex with OpenAI)"
        ) from e


def get_embedding_service_if_available() -> "EmbeddingServiceProtocol | None":
    """Safely attempt to load and initialize an embedding service.

    Unlike get_embedding_service(), this function returns None instead of
    raising ImportError when no service is available.

    Returns:
        EmbeddingServiceProtocol instance if dependencies are met, else None.
    """
    try:
        return get_embedding_service()
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


def get_analyzer_if_available() -> "StatisticalAnalyzer | None":
    """Safely attempt to load and initialize the StatisticalAnalyzer.

    Returns:
        StatisticalAnalyzer instance if Modal is available, else None.
    """
    try:
        from src.services.statistical_analyzer import get_statistical_analyzer

        analyzer = get_statistical_analyzer()
        logger.info("StatisticalAnalyzer initialized successfully")
        return analyzer
    except ImportError as e:
        logger.info(
            "StatisticalAnalyzer not available (Modal dependencies missing)",
            missing_dependency=str(e),
        )
    except Exception as e:
        logger.warning(
            "StatisticalAnalyzer initialization failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
        )
    return None
