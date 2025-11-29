"""Service loader utility for safe, lazy loading of optional services.

This module handles the import and initialization of services that may
have missing optional dependencies (like Modal or Sentence Transformers),
preventing the application from crashing if they are not available.
"""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService
    from src.services.statistical_analyzer import StatisticalAnalyzer

logger = structlog.get_logger()


def get_embedding_service_if_available() -> "EmbeddingService | None":
    """
    Safely attempt to load and initialize the EmbeddingService.

    Returns:
        EmbeddingService instance if dependencies are met, else None.
    """
    try:
        # Import here to avoid top-level dependency check
        from src.services.embeddings import get_embedding_service

        service = get_embedding_service()
        logger.info("Embedding service initialized successfully")
        return service
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
    """
    Safely attempt to load and initialize the StatisticalAnalyzer.

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
