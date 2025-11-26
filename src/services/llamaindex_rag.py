"""LlamaIndex RAG service for evidence retrieval and indexing.

Requires optional dependencies: uv sync --extra modal
"""

from typing import Any

import structlog

from src.utils.config import settings
from src.utils.models import Evidence

logger = structlog.get_logger()


class LlamaIndexRAGService:
    """RAG service using LlamaIndex with ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "deepcritical_evidence",
        persist_dir: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        similarity_top_k: int = 5,
    ) -> None:
        """
        Initialize LlamaIndex RAG service.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model to use
            similarity_top_k: Number of top results to retrieve
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
                "LlamaIndex dependencies not installed. Run: uv sync --extra modal"
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

        # Configure LlamaIndex settings (use centralized config)
        self._Settings.llm = OpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
        self._Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=settings.openai_api_key,
        )

        # Initialize ChromaDB client
        self.chroma_client = self._chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info("loaded_existing_collection", name=self.collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(self.collection_name)
            logger.info("created_new_collection", name=self.collection_name)

        # Initialize vector store and index
        self.vector_store = self._ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = self._StorageContext.from_defaults(vector_store=self.vector_store)

        # Try to load existing index, or create empty one
        try:
            self.index = self._VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )
            logger.info("loaded_existing_index")
        except Exception:
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
        except Exception as e:
            logger.error("failed_to_ingest_evidence", error=str(e))
            raise

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
        except Exception as e:
            logger.error("failed_to_ingest_documents", error=str(e))
            raise

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
                        "text": node.node.text,
                        "score": node.score,
                        "metadata": node.node.metadata,
                    }
                )

            logger.info("retrieved_documents", query=query[:50], count=len(results))
            return results

        except Exception as e:
            logger.error("failed_to_retrieve", error=str(e), query=query[:50])
            return []

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

        except Exception as e:
            logger.error("failed_to_query", error=str(e), query=query_str[:50])
            return f"Error generating response: {e}"

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
        except Exception as e:
            logger.error("failed_to_clear_collection", error=str(e))
            raise


def get_rag_service(
    collection_name: str = "deepcritical_evidence",
    **kwargs: Any,
) -> LlamaIndexRAGService:
    """
    Get or create a RAG service instance.

    Args:
        collection_name: Name of the ChromaDB collection
        **kwargs: Additional arguments for LlamaIndexRAGService

    Returns:
        Configured LlamaIndexRAGService instance
    """
    return LlamaIndexRAGService(collection_name=collection_name, **kwargs)
