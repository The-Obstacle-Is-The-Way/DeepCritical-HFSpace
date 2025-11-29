from unittest.mock import MagicMock, patch

from src.utils.service_loader import (
    get_analyzer_if_available,
    get_embedding_service_if_available,
)


def test_get_embedding_service_success():
    """Test successful loading of embedding service."""
    with patch("src.services.embeddings.get_embedding_service") as mock_get:
        mock_service = MagicMock()
        mock_get.return_value = mock_service

        service = get_embedding_service_if_available()

        assert service is mock_service
        mock_get.assert_called_once()


def test_get_embedding_service_import_error():
    """Test handling of ImportError when loading embedding service."""
    # Simulate import error by patching the function to raise ImportError
    with patch(
        "src.services.embeddings.get_embedding_service",
        side_effect=ImportError("Missing deps"),
    ):
        service = get_embedding_service_if_available()
        assert service is None


def test_get_embedding_service_generic_error():
    """Test handling of generic Exception when loading embedding service."""
    with patch(
        "src.services.embeddings.get_embedding_service",
        side_effect=ValueError("Boom"),
    ):
        service = get_embedding_service_if_available()
        assert service is None


def test_get_analyzer_success():
    """Test successful loading of analyzer."""
    with patch("src.services.statistical_analyzer.get_statistical_analyzer") as mock_get:
        mock_analyzer = MagicMock()
        mock_get.return_value = mock_analyzer

        analyzer = get_analyzer_if_available()

        assert analyzer is mock_analyzer
        mock_get.assert_called_once()


def test_get_analyzer_import_error():
    """Test handling of ImportError when loading analyzer."""
    with patch(
        "src.services.statistical_analyzer.get_statistical_analyzer",
        side_effect=ImportError("No Modal"),
    ):
        analyzer = get_analyzer_if_available()
        assert analyzer is None


def test_get_analyzer_generic_error():
    """Test handling of generic Exception when loading analyzer."""
    with patch(
        "src.services.statistical_analyzer.get_statistical_analyzer",
        side_effect=RuntimeError("Fail"),
    ):
        analyzer = get_analyzer_if_available()
        assert analyzer is None
