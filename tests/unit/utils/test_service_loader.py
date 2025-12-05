from unittest.mock import MagicMock, patch

from src.utils.service_loader import (
    get_embedding_service_if_available,
)


class TestGetEmbeddingServiceIfAvailable:
    """Test get_embedding_service_if_available() safety wrapper."""

    def test_returns_service_when_available(self):
        """Test successful loading of embedding service (free tier fallback)."""
        mock_service = MagicMock()

        # Patch settings to disable premium tier, then patch the local service
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch("src.services.embeddings.get_embedding_service", return_value=mock_service):
                service = get_embedding_service_if_available()
                assert service is mock_service

    def test_returns_none_when_no_service_available(self):
        """Test handling of ImportError when loading embedding service."""
        # Disable premium tier, then make local service fail
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch(
                "src.services.embeddings.get_embedding_service",
                side_effect=ImportError("Missing deps"),
            ):
                service = get_embedding_service_if_available()
                assert service is None


def test_get_embedding_service_generic_error():
    """Test handling of generic Exception when loading embedding service."""
    # Disable premium tier, then make local service fail
    with patch("src.utils.service_loader.settings") as mock_settings:
        mock_settings.has_openai_key = False

        with patch(
            "src.services.embeddings.get_embedding_service",
            side_effect=ValueError("Boom"),
        ):
            service = get_embedding_service_if_available()
            assert service is None


class TestGetEmbeddingService:
    """Test get_embedding_service() logic."""

    def test_uses_llamaindex_when_openai_key_present(self):
        """OpenAI key (env) → LlamaIndex."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True
            mock_settings.openai_api_key = "sk-env"

            # Mock LlamaIndex dependencies and factory
            with patch.dict(
                "sys.modules",
                {
                    "src.services.llamaindex_rag": MagicMock(),
                    "chromadb": MagicMock(),
                    "llama_index": MagicMock(),
                },
            ):
                mock_rag_service = MagicMock()
                with patch(
                    "src.services.llamaindex_rag.get_rag_service", return_value=mock_rag_service
                ):
                    from src.utils.service_loader import get_embedding_service

                    service = get_embedding_service()
                    assert service is mock_rag_service

    def test_uses_llamaindex_when_byok_key_present(self):
        """BYOK key → LlamaIndex."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch.dict(
                "sys.modules",
                {
                    "src.services.llamaindex_rag": MagicMock(),
                },
            ):
                mock_rag_service = MagicMock()
                with patch(
                    "src.services.llamaindex_rag.get_rag_service", return_value=mock_rag_service
                ):
                    from src.utils.service_loader import get_embedding_service

                    service = get_embedding_service(api_key="sk-test")
                    assert service is mock_rag_service

    def test_falls_back_to_local_when_no_openai_key(self):
        """No OpenAI key → Local embeddings."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            mock_local_service = MagicMock()
            with patch(
                "src.services.embeddings.get_embedding_service", return_value=mock_local_service
            ):
                from src.utils.service_loader import get_embedding_service

                service = get_embedding_service()
                assert service is mock_local_service

    def test_falls_back_when_llamaindex_import_fails(self):
        """LlamaIndex fails import → Local embeddings."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True

            # Mock ImportError for LlamaIndex
            with patch(
                "src.services.llamaindex_rag.get_rag_service", side_effect=ImportError("No deps")
            ):
                mock_local_service = MagicMock()
                with patch(
                    "src.services.embeddings.get_embedding_service", return_value=mock_local_service
                ):
                    from src.utils.service_loader import get_embedding_service

                    service = get_embedding_service()
                    assert service is mock_local_service

    def test_raises_when_no_embedding_service_available(self):
        """All services fail → ImportError."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch(
                "src.services.embeddings.get_embedding_service", side_effect=ImportError("No deps")
            ):
                import pytest

                from src.utils.service_loader import get_embedding_service

                with pytest.raises(ImportError) as exc:
                    get_embedding_service()
                assert "No embedding service available" in str(exc.value)
