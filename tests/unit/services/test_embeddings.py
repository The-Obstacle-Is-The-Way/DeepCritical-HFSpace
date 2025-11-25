"""Unit tests for EmbeddingService."""

from unittest.mock import patch

import numpy as np
import pytest

# Skip if embeddings dependencies are not installed
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from src.services.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("src.services.embeddings.SentenceTransformer") as mock_st_class:
            mock_model = mock_st_class.return_value
            # Mock encode to return a numpy array
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            yield mock_model

    @pytest.fixture
    def mock_chroma_client(self):
        with patch("src.services.embeddings.chromadb.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_collection = mock_client.create_collection.return_value
            # Mock query return structure
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "documents": [["doc1"]],
                "metadatas": [[{"source": "pubmed"}]],
                "distances": [[0.1]],
            }
            yield mock_client

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, mock_sentence_transformer, mock_chroma_client):
        """Embedding should return a float vector (async check)."""
        service = EmbeddingService()
        embedding = await service.embed("metformin diabetes")

        assert isinstance(embedding, list)
        assert len(embedding) == 3  # noqa: PLR2004
        assert all(isinstance(x, float) for x in embedding)
        # Ensure it ran in executor (mock encode called)
        mock_sentence_transformer.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_embed_efficient(self, mock_sentence_transformer, mock_chroma_client):
        """Batch embedding should call encode with list."""
        # Setup mock for batch return (list of arrays)
        import numpy as np

        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        service = EmbeddingService()
        texts = ["text one", "text two"]

        batch_results = await service.embed_batch(texts)

        assert len(batch_results) == 2  # noqa: PLR2004
        assert isinstance(batch_results[0], list)
        mock_sentence_transformer.encode.assert_called_with(texts)

    @pytest.mark.asyncio
    async def test_add_and_search(self, mock_sentence_transformer, mock_chroma_client):
        """Should be able to add evidence and search for similar."""
        service = EmbeddingService()
        await service.add_evidence(
            evidence_id="test1",
            content="Metformin activates AMPK pathway",
            metadata={"source": "pubmed"},
        )

        # Verify add was called
        mock_collection = mock_chroma_client.create_collection.return_value
        mock_collection.add.assert_called_once()

        results = await service.search_similar("AMPK activation drugs", n_results=1)

        # Verify query was called
        mock_collection.query.assert_called_once()
        assert len(results) == 1
        assert results[0]["id"] == "id1"

    @pytest.mark.asyncio
    async def test_search_similar_empty_collection(
        self, mock_sentence_transformer, mock_chroma_client
    ):
        """Search on empty collection should return empty list, not error."""
        mock_collection = mock_chroma_client.create_collection.return_value
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        service = EmbeddingService()
        results = await service.search_similar("anything", n_results=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_deduplicate(self, mock_sentence_transformer, mock_chroma_client):
        """Deduplicate should remove similar items."""
        from src.utils.models import Citation, Evidence

        service = EmbeddingService()

        # Mock search to return a match for the first item (duplicate)
        # and no match for the second (unique)
        mock_collection = mock_chroma_client.create_collection.return_value

        # First call returns match (distance 0.05 < threshold)
        # Second call returns no match or high distance
        mock_collection.query.side_effect = [
            {
                "ids": [["existing_id"]],
                "documents": [["doc"]],
                "metadatas": [[{}]],
                "distances": [[0.05]],  # Very similar
            },
            {
                "ids": [[]],  # No match
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            },
        ]

        evidence = [
            Evidence(
                content="Duplicate content",
                citation=Citation(source="web", url="u1", title="t1", date="2024"),
            ),
            Evidence(
                content="Unique content",
                citation=Citation(source="web", url="u2", title="t2", date="2024"),
            ),
        ]

        unique = await service.deduplicate(evidence, threshold=0.9)

        # Only the unique one should remain
        assert len(unique) == 1
        assert unique[0].citation.url == "u2"
