"""Unit tests for text utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.models import Citation, Evidence
from src.utils.text_utils import select_diverse_evidence, truncate_at_sentence


class TestTextUtils:
    def test_truncate_at_sentence_short(self):
        """Should return text as is if shorter than limit."""
        text = "This is a short sentence."
        assert truncate_at_sentence(text, 100) == text

    def test_truncate_at_sentence_boundary(self):
        """Should truncate at sentence ending."""
        text = "First sentence. Second sentence. Third sentence."
        # Limit should cut in the middle of second sentence
        limit = len("First sentence. Second sentence") + 5
        result = truncate_at_sentence(text, limit)
        assert result == "First sentence. Second sentence."

    def test_truncate_at_sentence_fallback_period(self):
        """Should fall back to period if no sentence boundary found."""
        text = "Dr. Smith went to the store. He bought apples."
        # Limit cuts in "He bought"
        limit = len("Dr. Smith went to the store.") + 5
        result = truncate_at_sentence(text, limit)
        assert result == "Dr. Smith went to the store."

    def test_truncate_at_sentence_fallback_word(self):
        """Should fall back to word boundary if no punctuation."""
        text = "This is a very long sentence without any punctuation marks until the very end"
        limit = 20
        result = truncate_at_sentence(text, limit)
        assert result == "This is a very long..."
        # Ellipsis might add length, checking logic
        assert len(result) <= limit + 3  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_select_diverse_evidence_no_embeddings(self):
        """Should fallback to relevance sort if no embeddings."""
        evidence = [
            Evidence(
                content="A",
                relevance=0.9,
                citation=Citation(source="pubmed", title="A", url="a", date="2023"),
            ),
            Evidence(
                content="B",
                relevance=0.1,
                citation=Citation(source="pubmed", title="B", url="b", date="2023"),
            ),
            Evidence(
                content="C",
                relevance=0.8,
                citation=Citation(source="pubmed", title="C", url="c", date="2023"),
            ),
        ]

        selected = await select_diverse_evidence(evidence, n=2, query="test", embeddings=None)

        expected_count = 2
        assert len(selected) == expected_count
        assert selected[0].content == "A"  # Highest relevance
        assert selected[1].content == "C"  # Second highest

    @pytest.mark.asyncio
    async def test_select_diverse_evidence_mmr(self):
        """Should select diverse evidence using MMR."""
        # Mock embeddings
        mock_embeddings = MagicMock()

        # Scenario: Query is equidistant to A and C.
        # A and B are identical (clones).
        # C is orthogonal to A/B.
        # We expect A (first) then C (diverse), skipping B (clone).

        # Query: [0.707, 0.707]
        # A: [1.0, 0.0] -> Sim to Q: 0.707
        # B: [1.0, 0.0] -> Sim to Q: 0.707, Sim to A: 1.0
        # C: [0.0, 1.0] -> Sim to Q: 0.707, Sim to A: 0.0

        async def mock_embed(text):
            if text == "query":
                return [0.707, 0.707]
            return [0.0, 0.0]

        async def mock_embed_batch(texts):
            results = []
            for t in texts:
                if t == "A":
                    results.append([1.0, 0.0])
                elif t == "B":
                    results.append([1.0, 0.0])  # Clone of A
                elif t == "C":
                    results.append([0.0, 1.0])  # Orthogonal
                else:
                    results.append([0.0, 0.0])
            return results

        mock_embeddings.embed = AsyncMock(side_effect=mock_embed)
        mock_embeddings.embed_batch = AsyncMock(side_effect=mock_embed_batch)

        evidence = [
            Evidence(
                content="A",
                relevance=0.9,
                citation=Citation(source="pubmed", title="A", url="a", date="2023"),
            ),
            Evidence(
                content="B",
                relevance=0.9,
                citation=Citation(source="pubmed", title="B", url="b", date="2023"),
            ),
            Evidence(
                content="C",
                relevance=0.9,
                citation=Citation(source="pubmed", title="C", url="c", date="2023"),
            ),
        ]

        # With n=2, we expect A then C.
        selected = await select_diverse_evidence(
            evidence, n=2, query="query", embeddings=mock_embeddings
        )

        expected_count = 2
        assert len(selected) == expected_count
        assert selected[0].content == "A"
        assert selected[1].content == "C"
