"""LlamaIndex RAG service for evidence retrieval and indexing.

Requires optional dependencies: uv sync --extra rag

Migration Note (v1.0 rebrand):
    Default collection_name changed from "deepcritical_evidence" to "deepboner_evidence".
    To preserve existing data, explicitly pass collection_name="deepcritical_evidence".

Protocol Compliance:
    This service implements EmbeddingServiceProtocol via async wrapper methods:
    - add_evidence() - async wrapper for ingest_evidence()
    - search_similar() - async wrapper for retrieve()
    - deduplicate() - async wrapper using search_similar() + add_evidence()

    These wrappers use asyncio.run_in_executor() to avoid blocking the event loop.
"""

import asyncio
from typing import Any

import structlog

from src.utils.config import settings
from src.utils.exceptions import ConfigurationError, EmbeddingError
from src.utils.models import Citation, Evidence

logger = structlog.get_logger()


class LlamaIndexRAGService:
    """RAG service using LlamaIndex with ChromaDB vector store.

    Note:
        This service is currently OpenAI-only. It uses OpenAI embeddings and LLM
        regardless of the global `settings.llm_provider` configuration.
        Requires OPENAI_API_KEY to be set.
    """

    def __init__(
        self,
        collection_name: str = "deepboner_evidence",
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        similarity_top_k: int = 5,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize LlamaIndex RAG service.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model (defaults to settings.openai_embedding_model)
            similarity_top_k: Number of top results to retrieve
            api_key: Optional BYOK OpenAI key. Prioritized over env var.
        """
        # Lazy import - only when instantiated
        try:
            import chromadb
            from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.llms.openai import OpenAI
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError as e:
            raise ImportError(
                "LlamaIndex dependencies not installed. Run: uv sync --extra rag"
            ) from e

        # Store references for use in other methods
        self._chromadb = chromadb
        self._Document = Document
        self._Settings = Settings
        self._StorageContext = StorageContext
        self._VectorStoreIndex = VectorStoreIndex
        self._VectorIndexRetriever = VectorIndexRetriever
        self._ChromaVectorStore = ChromaVectorStore

        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.chroma_db_path
        self.similarity_top_k = similarity_top_k
        self.embedding_model = embedding_model or settings.openai_embedding_model

        # Determine API key (BYOK > Env Var)
        self.api_key = api_key
        if not self.api_key and settings.has_openai_key:
            self.api_key = settings.openai_api_key

        # Validate API key before use
        if not self.api_key:
            raise ConfigurationError("OPENAI_API_KEY required for LlamaIndex RAG service")

        # Defense-in-depth: Validate key prefix to prevent cryptic auth errors
        if not self.api_key.startswith("sk-"):
            raise ConfigurationError(
                f"Invalid API key format. Expected OpenAI key starting with 'sk-', "
                f"got key starting with '{self.api_key[:8]}...'."
            )

        # Configure LlamaIndex settings (use centralized config)
        self._Settings.llm = OpenAI(
            model=settings.openai_model,
            api_key=self.api_key,
        )
        self._Settings.embed_model = OpenAIEmbedding(
            model=self.embedding_model,
            api_key=self.api_key,
        )

        # Initialize ChromaDB client
        self.chroma_client = self._chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        # ChromaDB raises different exceptions depending on version:
        # - ValueError (older versions)
        # - InvalidCollectionException / NotFoundError (newer versions)
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info("loaded_existing_collection", name=self.collection_name)
        except Exception as e:
            # Catch any collection-not-found error and create it
            if (
                "not exist" in str(e).lower()
                or "not found" in str(e).lower()
                or isinstance(e, ValueError)
            ):
                self.collection = self.chroma_client.create_collection(self.collection_name)
                logger.info("created_new_collection", name=self.collection_name)
            else:
                raise

        # Initialize vector store and index
        self.vector_store = self._ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = self._StorageContext.from_defaults(vector_store=self.vector_store)

        # Try to load existing index, or create empty one
        # LlamaIndex raises ValueError for empty/invalid stores
        try:
            self.index = self._VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )
            logger.info("loaded_existing_index")
        except (ValueError, KeyError):
            # Empty or newly created store - create fresh index
            self.index = self._VectorStoreIndex([], storage_context=self.storage_context)
            logger.info("created_new_index")

    def ingest_evidence(self, evidence_list: list[Evidence]) -> None:
        """
        Ingest evidence into the vector store.

        Args:
            evidence_list: List of Evidence objects to ingest
        """
        if not evidence_list:
            logger.warning("no_evidence_to_ingest")
            return

        # Convert Evidence objects to LlamaIndex Documents
        documents = []
        for evidence in evidence_list:
            metadata = {
                "source": evidence.citation.source,
                "title": evidence.citation.title,
                "url": evidence.citation.url,
                "date": evidence.citation.date,
                "authors": ", ".join(evidence.citation.authors),
            }

            doc = self._Document(
                text=evidence.content,
                metadata=metadata,
                doc_id=evidence.citation.url,  # Use URL as unique ID
            )
            documents.append(doc)

        # Insert documents into index
        try:
            for doc in documents:
                self.index.insert(doc)
            logger.info("ingested_evidence", count=len(documents))
        except (ValueError, RuntimeError) as e:
            logger.error("failed_to_ingest_evidence", error=str(e))
            raise EmbeddingError(f"Failed to ingest evidence: {e}") from e

    def ingest_documents(self, documents: list[Any]) -> None:
        """
        Ingest raw LlamaIndex Documents.

        Args:
            documents: List of LlamaIndex Document objects
        """
        if not documents:
            logger.warning("no_documents_to_ingest")
            return

        try:
            for doc in documents:
                self.index.insert(doc)
            logger.info("ingested_documents", count=len(documents))
        except (ValueError, RuntimeError) as e:
            logger.error("failed_to_ingest_documents", error=str(e))
            raise EmbeddingError(f"Failed to ingest documents: {e}") from e

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of results to return (defaults to similarity_top_k)

        Returns:
            List of retrieved documents with metadata and scores
        """
        k = top_k or self.similarity_top_k

        # Create retriever
        retriever = self._VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )

        try:
            # Retrieve nodes
            nodes = retriever.retrieve(query)

            # Convert to dict format
            results = []
            for node in nodes:
                results.append(
                    {
                        "text": node.node.get_content(),
                        "score": node.score,
                        "metadata": node.node.metadata,
                    }
                )

            logger.info("retrieved_documents", query=query[:50], count=len(results))
            return results

        except (ValueError, RuntimeError) as e:
            logger.error("failed_to_retrieve", error=str(e), query=query[:50])
            raise EmbeddingError(f"Failed to retrieve documents: {e}") from e

    def query(self, query_str: str, top_k: int | None = None) -> str:
        """
        Query the RAG system and get a synthesized response.

        Args:
            query_str: Query string
            top_k: Number of results to use (defaults to similarity_top_k)

        Returns:
            Synthesized response string
        """
        k = top_k or self.similarity_top_k

        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=k,
        )

        try:
            response = query_engine.query(query_str)
            logger.info("generated_response", query=query_str[:50])
            return str(response)

        except (ValueError, RuntimeError) as e:
            logger.error("failed_to_query", error=str(e), query=query_str[:50])
            raise EmbeddingError(f"Failed to query RAG system: {e}") from e

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(self.collection_name)
            self.vector_store = self._ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = self._StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = self._VectorStoreIndex([], storage_context=self.storage_context)
            logger.info("cleared_collection", name=self.collection_name)
        except (ValueError, RuntimeError) as e:
            logger.error("failed_to_clear_collection", error=str(e))
            raise EmbeddingError(f"Failed to clear collection: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # Async Protocol Methods (EmbeddingServiceProtocol compliance)
    # ─────────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single text using OpenAI embeddings (Protocol-compatible).

        Uses the LlamaIndex Settings.embed_model which was configured in __init__.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        loop = asyncio.get_running_loop()
        # LlamaIndex embed_model has get_text_embedding method
        embedding = await loop.run_in_executor(
            None, self._Settings.embed_model.get_text_embedding, text
        )
        return list(embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently (Protocol-compatible).

        Uses LlamaIndex's batch embedding for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        # LlamaIndex embed_model has get_text_embedding_batch method
        embeddings = await loop.run_in_executor(
            None, self._Settings.embed_model.get_text_embedding_batch, texts
        )
        return [list(emb) for emb in embeddings]

    async def add_evidence(self, evidence_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Async wrapper for adding evidence (Protocol-compatible).

        Converts the sync ingest_evidence pattern to the async protocol interface.
        Uses run_in_executor to avoid blocking the event loop.

        Args:
            evidence_id: Unique identifier (typically URL)
            content: Text content to embed and store
            metadata: Additional metadata (source, title, date, authors)
        """
        # Reconstruct Evidence from parts
        authors_str = metadata.get("authors", "")
        authors = [a.strip() for a in authors_str.split(",")] if authors_str else []

        citation = Citation(
            source=metadata.get("source", "web"),
            title=metadata.get("title", "Unknown"),
            url=evidence_id,
            date=metadata.get("date", "Unknown"),
            authors=authors,
        )
        evidence = Evidence(content=content, citation=citation)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.ingest_evidence, [evidence])

    async def search_similar(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Async wrapper for retrieve (Protocol-compatible).

        Returns results in the same format as EmbeddingService.search_similar()
        for seamless interchangeability.

        Args:
            query: Search query text
            n_results: Maximum number of results to return

        Returns:
            List of dicts with keys: id, content, metadata, distance
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.retrieve, query, n_results)

        # Convert LlamaIndex format to EmbeddingService format for compatibility
        # LlamaIndex: {"text": ..., "score": ..., "metadata": ...}
        # EmbeddingService: {"id": ..., "content": ..., "metadata": ..., "distance": ...}
        return [
            {
                "id": r.get("metadata", {}).get("url", ""),
                "content": r.get("text", ""),
                "metadata": r.get("metadata", {}),
                # Convert similarity score to distance
                # LlamaIndex score: 0-1 (higher = more similar)
                # Output distance: 0-1 (lower = more similar, matches ChromaDB behavior)
                "distance": 1.0 - r.get("score", 0.5),
            }
            for r in results
        ]

    async def deduplicate(self, evidence: list[Evidence], threshold: float = 0.9) -> list[Evidence]:
        """Async wrapper for deduplication (Protocol-compatible).

        Uses search_similar() to check for existing similar content.
        Stores unique evidence and returns the deduplicated list.

        Args:
            evidence: List of evidence items to deduplicate
            threshold: Similarity threshold (0.9 = 90% similar is duplicate)
                Distance range: 0-1 (0 = identical, 1 = orthogonal)
                Duplicate if: distance < (1 - threshold), e.g., < 0.1 for 90%

        Returns:
            List of unique evidence items (duplicates removed)
        """
        unique = []

        for ev in evidence:
            try:
                # Check for similar existing content
                similar = await self.search_similar(ev.content, n_results=1)

                # Check similarity threshold
                # distance 0 = identical, higher = more different
                is_duplicate = similar and similar[0]["distance"] < (1 - threshold)

                if not is_duplicate:
                    unique.append(ev)
                    # Store the new evidence
                    await self.add_evidence(
                        evidence_id=ev.citation.url,
                        content=ev.content,
                        metadata={
                            "source": ev.citation.source,
                            "title": ev.citation.title,
                            "date": ev.citation.date,
                            "authors": ",".join(ev.citation.authors or []),
                        },
                    )
            except Exception as e:
                # Log but don't fail - better to have duplicates than lose data
                logger.warning(
                    "Failed to process evidence in deduplicate",
                    url=ev.citation.url,
                    error=str(e),
                )
                unique.append(ev)

        return unique


def get_rag_service(
    collection_name: str = "deepboner_evidence",
    api_key: str | None = None,
    **kwargs: Any,
) -> LlamaIndexRAGService:
    """
    Get or create a RAG service instance.

    Args:
        collection_name: Name of the ChromaDB collection
        api_key: Optional BYOK OpenAI key
        **kwargs: Additional arguments for LlamaIndexRAGService

    Returns:
        Configured LlamaIndexRAGService instance
    """
    return LlamaIndexRAGService(collection_name=collection_name, api_key=api_key, **kwargs)
