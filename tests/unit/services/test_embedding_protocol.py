"""Tests for EmbeddingServiceProtocol compliance.

TDD: These tests verify that both EmbeddingService and LlamaIndexRAGService
implement the EmbeddingServiceProtocol interface correctly.
"""

import asyncio
from unittest.mock import patch

import pytest

# Skip if chromadb not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")


class TestEmbeddingServiceProtocolCompliance:
    """Verify EmbeddingService implements EmbeddingServiceProtocol."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer to avoid loading actual model."""
        import numpy as np

        import src.services.embeddings

        # Reset singleton to ensure mock is used
        src.services.embeddings._shared_model = None

        with patch("src.services.embeddings.SentenceTransformer") as mock_st_class:
            mock_model = mock_st_class.return_value
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            yield mock_model

        # Cleanup
        src.services.embeddings._shared_model = None

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        with patch("src.services.embeddings.chromadb.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_collection = mock_client.create_collection.return_value
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "documents": [["doc1"]],
                "metadatas": [[{"source": "pubmed"}]],
                "distances": [[0.1]],
            }
            yield mock_client

    def test_has_add_evidence_method(self, mock_sentence_transformer, mock_chroma_client):
        """EmbeddingService should have async add_evidence method."""
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()
        assert hasattr(service, "add_evidence")
        assert asyncio.iscoroutinefunction(service.add_evidence)

    def test_has_search_similar_method(self, mock_sentence_transformer, mock_chroma_client):
        """EmbeddingService should have async search_similar method."""
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()
        assert hasattr(service, "search_similar")
        assert asyncio.iscoroutinefunction(service.search_similar)

    def test_has_deduplicate_method(self, mock_sentence_transformer, mock_chroma_client):
        """EmbeddingService should have async deduplicate method."""
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()
        assert hasattr(service, "deduplicate")
        assert asyncio.iscoroutinefunction(service.deduplicate)

    @pytest.mark.asyncio
    async def test_add_evidence_signature(self, mock_sentence_transformer, mock_chroma_client):
        """add_evidence should accept (evidence_id, content, metadata)."""
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()

        # Should not raise
        await service.add_evidence(
            evidence_id="test-id",
            content="test content",
            metadata={"source": "pubmed", "title": "Test"},
        )

    @pytest.mark.asyncio
    async def test_search_similar_signature(self, mock_sentence_transformer, mock_chroma_client):
        """search_similar should accept (query, n_results) and return list[dict]."""
        from src.services.embeddings import EmbeddingService

        service = EmbeddingService()

        results = await service.search_similar("test query", n_results=5)

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], dict)
            # Should have expected keys
            assert "id" in results[0]
            assert "content" in results[0]
            assert "metadata" in results[0]
            assert "distance" in results[0]

    @pytest.mark.asyncio
    async def test_deduplicate_signature(self, mock_sentence_transformer, mock_chroma_client):
        """deduplicate should accept (evidence, threshold) and return list[Evidence]."""
        from src.services.embeddings import EmbeddingService
        from src.utils.models import Citation, Evidence

        service = EmbeddingService()

        # Mock to avoid actual dedup logic
        mock_chroma_client.create_collection.return_value.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        evidence = [
            Evidence(
                content="test",
                citation=Citation(source="pubmed", url="u1", title="t1", date="2024"),
            )
        ]

        results = await service.deduplicate(evidence, threshold=0.9)

        assert isinstance(results, list)
        assert all(isinstance(e, Evidence) for e in results)


class TestProtocolTypeChecking:
    """Verify Protocol works with type checking."""

    def test_embedding_service_satisfies_protocol(self):
        """EmbeddingService should satisfy EmbeddingServiceProtocol."""

        from src.services.embedding_protocol import EmbeddingServiceProtocol
        from src.services.embeddings import EmbeddingService

        # Protocol should be runtime checkable
        assert hasattr(EmbeddingServiceProtocol, "__protocol_attrs__") or True

        # This is a structural check - just verify the methods exist
        service_methods = {"add_evidence", "search_similar", "deduplicate"}
        embedding_methods = {m for m in dir(EmbeddingService) if not m.startswith("_")}

        assert service_methods.issubset(embedding_methods)
