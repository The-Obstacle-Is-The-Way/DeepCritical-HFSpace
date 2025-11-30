"""Tests for service loader embedding service selection.

TDD: These tests define the expected behavior of get_embedding_service().
"""

from unittest.mock import MagicMock, patch

import pytest


class TestGetEmbeddingService:
    """Tests for get_embedding_service() tiered selection."""

    def test_uses_llamaindex_when_openai_key_present(self):
        """Should return LlamaIndexRAGService when OPENAI_API_KEY is set."""
        mock_rag_service = MagicMock()

        # Patch at the point of use (inside service_loader)
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True

            with patch(
                "src.utils.service_loader.get_rag_service",
                return_value=mock_rag_service,
                create=True,
            ):
                # Also need to prevent the actual import from failing
                mock_module = MagicMock(get_rag_service=lambda: mock_rag_service)
                with patch.dict("sys.modules", {"src.services.llamaindex_rag": mock_module}):
                    from src.utils.service_loader import get_embedding_service

                    service = get_embedding_service()
                    assert service is mock_rag_service

    def test_falls_back_to_local_when_no_openai_key(self):
        """Should return EmbeddingService when no OpenAI key."""
        mock_local_service = MagicMock()

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            # Patch the embeddings module
            mock_embed_mod = MagicMock(get_embedding_service=lambda: mock_local_service)
            with patch.dict("sys.modules", {"src.services.embeddings": mock_embed_mod}):
                from src.utils.service_loader import get_embedding_service

                service = get_embedding_service()
                assert service is mock_local_service

    def test_falls_back_when_llamaindex_import_fails(self):
        """Should fallback to local if LlamaIndex deps missing."""
        mock_local_service = MagicMock()

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = True

            # LlamaIndex import fails
            def raise_import_error(*args, **kwargs):
                raise ImportError("llama_index not installed")

            # Make llamaindex_rag module raise ImportError on import
            import sys
            original_modules = dict(sys.modules)

            # Remove llamaindex_rag if it exists
            if "src.services.llamaindex_rag" in sys.modules:
                del sys.modules["src.services.llamaindex_rag"]

            try:
                # Patch to raise ImportError
                mock_embed_module = MagicMock(
                    get_embedding_service=lambda: mock_local_service
                )
                with patch.dict(
                    "sys.modules",
                    {
                        "src.services.llamaindex_rag": None,  # None causes ImportError
                        "src.services.embeddings": mock_embed_module,
                    },
                ):
                    from src.utils.service_loader import get_embedding_service

                    service = get_embedding_service()
                    assert service is mock_local_service
            finally:
                # Restore original modules
                sys.modules.update(original_modules)

    def test_raises_when_no_embedding_service_available(self):
        """Should raise ImportError when no embedding service can be loaded."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            # Make embeddings module raise ImportError
            with patch.dict(
                "sys.modules",
                {"src.services.embeddings": None},  # None causes ImportError
            ):
                from src.utils.service_loader import get_embedding_service

                with pytest.raises(ImportError) as exc_info:
                    get_embedding_service()

                assert "No embedding service available" in str(exc_info.value)


class TestGetEmbeddingServiceIfAvailable:
    """Tests for get_embedding_service_if_available() safe wrapper."""

    def test_returns_none_when_no_service_available(self):
        """Should return None instead of raising when no service available."""
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            # Make embeddings module raise ImportError
            with patch.dict(
                "sys.modules",
                {"src.services.embeddings": None},
            ):
                from src.utils.service_loader import get_embedding_service_if_available

                result = get_embedding_service_if_available()
                assert result is None

    def test_returns_service_when_available(self):
        """Should return the service when available."""
        mock_service = MagicMock()

        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            with patch.dict(
                "sys.modules",
                {"src.services.embeddings": MagicMock(get_embedding_service=lambda: mock_service)},
            ):
                from src.utils.service_loader import get_embedding_service_if_available

                result = get_embedding_service_if_available()
                assert result is mock_service
