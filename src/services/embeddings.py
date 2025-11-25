"""Embedding service for semantic search.

IMPORTANT: All public methods are async to avoid blocking the event loop.
The sentence-transformers model is CPU-bound, so we use run_in_executor().
"""

import asyncio
from typing import Any

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from src.utils.models import Evidence


class EmbeddingService:
    """Handles text embedding and vector storage.

    All embedding operations run in a thread pool to avoid blocking
    the async event loop. See src/tools/websearch.py for the pattern.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._client = chromadb.Client()  # In-memory for hackathon
        self._collection = self._client.create_collection(
            name="evidence", metadata={"hnsw:space": "cosine"}
        )

    # ─────────────────────────────────────────────────────────────────
    # Sync internal methods (run in thread pool)
    # ─────────────────────────────────────────────────────────────────

    def _sync_embed(self, text: str) -> list[float]:
        """Synchronous embedding - DO NOT call directly from async code."""
        result: list[float] = self._model.encode(text).tolist()
        return result

    def _sync_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding for efficiency - DO NOT call directly from async code."""
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    # ─────────────────────────────────────────────────────────────────
    # Async public methods (safe for event loop)
    # ─────────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single text (async-safe).

        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_embed, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts (async-safe, more efficient)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_batch_embed, texts)

    async def add_evidence(self, evidence_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Add evidence to vector store (async-safe)."""
        embedding = await self.embed(content)
        # ChromaDB operations are fast, but wrap for consistency
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[evidence_id],
                embeddings=[embedding],  # type: ignore[arg-type]
                metadatas=[metadata],
                documents=[content],
            ),
        )

    async def search_similar(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Find semantically similar evidence (async-safe)."""
        query_embedding = await self.embed(query)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=n_results,
            ),
        )

        # Handle empty results gracefully
        ids = results.get("ids")
        docs = results.get("documents")
        metas = results.get("metadatas")
        dists = results.get("distances")

        if not ids or not ids[0] or not docs or not metas or not dists:
            return []

        return [
            {"id": id, "content": doc, "metadata": meta, "distance": dist}
            for id, doc, meta, dist in zip(
                ids[0],
                docs[0],
                metas[0],
                dists[0],
                strict=False,
            )
        ]

    async def deduplicate(
        self, new_evidence: list[Evidence], threshold: float = 0.9
    ) -> list[Evidence]:
        """Remove semantically duplicate evidence (async-safe).

        Args:
            new_evidence: List of evidence items to deduplicate
            threshold: Similarity threshold (0.9 = 90% similar is duplicate).
                      ChromaDB cosine distance: 0=identical, 2=opposite.
                      We consider duplicate if distance < (1 - threshold).

        Returns:
            List of unique evidence items (not already in vector store).
        """
        unique = []
        for evidence in new_evidence:
            try:
                similar = await self.search_similar(evidence.content, n_results=1)
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # threshold=0.9 means distance < 0.1 is considered duplicate
                is_duplicate = similar and similar[0]["distance"] < (1 - threshold)

                if not is_duplicate:
                    unique.append(evidence)
                    # Store FULL citation metadata for reconstruction later
                    await self.add_evidence(
                        evidence_id=evidence.citation.url,
                        content=evidence.content,
                        metadata={
                            "source": evidence.citation.source,
                            "title": evidence.citation.title,
                            "date": evidence.citation.date,
                            "authors": ",".join(evidence.citation.authors or []),
                        },
                    )
            except Exception as e:
                # Log but don't fail entire deduplication for one bad item
                structlog.get_logger().warning(
                    "Failed to process evidence in deduplicate",
                    url=evidence.citation.url,
                    error=str(e),
                )
                # Still add to unique list - better to have duplicates than lose data
                unique.append(evidence)

        return unique


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton instance of EmbeddingService."""
    global _embedding_service  # noqa: PLW0603
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
