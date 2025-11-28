"""Integration tests for RAG integration.

These tests use HuggingFace (default) and may make real API calls.
Marked with @pytest.mark.integration and @pytest.mark.huggingface.
"""

import asyncio

import pytest

from src.services.llamaindex_rag import get_rag_service
from src.tools.rag_tool import create_rag_tool
from src.tools.search_handler import SearchHandler
from src.tools.tool_executor import execute_agent_task
from src.utils.config import settings
from src.utils.models import AgentTask, Citation, Evidence


@pytest.mark.integration
@pytest.mark.local_embeddings
class TestRAGServiceIntegration:
    """Integration tests for LlamaIndexRAGService (using HuggingFace)."""

    @pytest.mark.asyncio
    async def test_rag_service_ingest_and_retrieve(self):
        """RAG service should ingest and retrieve evidence."""
        # HuggingFace works without API key for public models
        # Use HuggingFace embeddings (default)
        rag_service = get_rag_service(
            collection_name="test_integration",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )

        # Create sample evidence
        evidence_list = [
            Evidence(
                content="Metformin is a first-line treatment for type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity.",
                citation=Citation(
                    source="pubmed",
                    title="Metformin Mechanism of Action",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    date="2024-01-15",
                    authors=["Smith J", "Johnson M"],
                ),
                relevance=0.9,
            ),
            Evidence(
                content="Recent studies suggest metformin may have neuroprotective effects in Alzheimer's disease models.",
                citation=Citation(
                    source="pubmed",
                    title="Metformin and Neuroprotection",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345679/",
                    date="2024-02-20",
                    authors=["Brown K", "Davis L"],
                ),
                relevance=0.85,
            ),
        ]

        # Ingest evidence
        rag_service.ingest_evidence(evidence_list)

        # Retrieve evidence
        results = rag_service.retrieve("metformin diabetes", top_k=2)

        # Assert
        assert len(results) > 0
        assert any("metformin" in r["text"].lower() for r in results)
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_rag_service_query(self):
        """RAG service should synthesize responses from ingested evidence."""
        # Require HF_TOKEN for query synthesis (LLM is needed)
        if not settings.has_huggingface_key:
            pytest.skip("HF_TOKEN required for HuggingFace LLM query synthesis")
        # Use HuggingFace LLM for query synthesis (default)
        rag_service = get_rag_service(
            collection_name="test_query",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )

        # Ingest evidence
        evidence_list = [
            Evidence(
                content="Python is a high-level programming language known for its simplicity and readability.",
                citation=Citation(
                    source="pubmed",
                    title="Python Programming",
                    url="https://example.com/python",
                    date="2024",
                    authors=["Author"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Check if LLM is available (might fail if model not available via inference API)
        if not rag_service._Settings.llm:
            pytest.skip(
                "HuggingFace LLM not available - model may not be accessible via inference API"
            )

        # Query with timeout
        # Note: query() is synchronous, but we wrap it to prevent hanging
        # If it takes too long, we'll get a timeout
        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: rag_service.query("What is Python?", top_k=1)),
                timeout=120.0,  # 2 minute timeout
            )

            assert isinstance(response, str)
            assert len(response) > 0
            assert "python" in response.lower()
        except Exception as e:
            # If model is not available (404), skip the test
            if "404" in str(e) or "Not Found" in str(e):
                pytest.skip(f"HuggingFace model not available via inference API: {e}")
            raise

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
@pytest.mark.local_embeddings
class TestRAGToolIntegration:
    """Integration tests for RAGTool (using HuggingFace)."""

    @pytest.mark.asyncio
    async def test_rag_tool_search(self):
        """RAGTool should search RAG service and return Evidence objects."""
        # HuggingFace works without API key for public models
        # Create RAG service and ingest evidence
        rag_service = get_rag_service(
            collection_name="test_rag_tool",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        evidence_list = [
            Evidence(
                content="Machine learning is a subset of artificial intelligence.",
                citation=Citation(
                    source="pubmed",
                    title="ML Basics",
                    url="https://example.com/ml",
                    date="2024",
                    authors=["ML Expert"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Create RAG tool
        tool = create_rag_tool(rag_service=rag_service)

        # Search
        results = await tool.search("machine learning", max_results=5)

        # Assert
        assert len(results) > 0
        assert all(isinstance(e, Evidence) for e in results)
        assert results[0].citation.source == "rag"
        assert (
            "machine learning" in results[0].content.lower()
            or "artificial intelligence" in results[0].content.lower()
        )

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_rag_tool_empty_collection(self):
        """RAGTool should return empty list when collection is empty."""
        # HuggingFace works without API key for public models
        rag_service = get_rag_service(
            collection_name="test_empty",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        rag_service.clear_collection()  # Ensure empty

        tool = create_rag_tool(rag_service=rag_service)
        results = await tool.search("any query")

        assert results == []


@pytest.mark.integration
@pytest.mark.local_embeddings
class TestRAGAgentIntegration:
    """Integration tests for RAGAgent in tool executor (using HuggingFace)."""

    @pytest.mark.asyncio
    async def test_rag_agent_execution(self):
        """RAGAgent should execute and return ToolAgentOutput."""
        # Require HF_TOKEN for query synthesis (LLM is needed for RAG query)
        if not settings.has_huggingface_key:
            pytest.skip("HF_TOKEN required for HuggingFace LLM query synthesis")
        # Setup: Ingest evidence into RAG
        rag_service = get_rag_service(
            collection_name="test_rag_agent",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        evidence_list = [
            Evidence(
                content="Deep learning uses neural networks with multiple layers. Neural networks are computational models inspired by biological neural networks.",
                citation=Citation(
                    source="pubmed",
                    title="Deep Learning",
                    url="https://example.com/dl",
                    date="2024",
                    authors=["DL Researcher"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Create RAG tool with the same service instance to ensure same collection
        from src.tools.rag_tool import RAGTool

        rag_tool = RAGTool(rag_service=rag_service)

        # Manually inject the RAG tool into the executor
        # Since execute_agent_task uses a module-level RAG tool, we need to patch it
        from unittest.mock import patch

        from src.tools import tool_executor

        # Patch the module-level _rag_tool variable
        with patch.object(tool_executor, "_rag_tool", rag_tool):
            # Execute RAGAgent task with timeout
            task = AgentTask(
                agent="RAGAgent",
                query="deep learning",
                gap="Need information about deep learning",
            )

            result = await asyncio.wait_for(
                execute_agent_task(task),
                timeout=120.0,  # 2 minute timeout
            )

        # Assert
        assert result.output
        # Check that the output contains relevant content (either from our evidence or general RAG results)
        output_lower = result.output.lower()
        has_relevant_content = (
            "deep learning" in output_lower
            or "neural network" in output_lower
            or "neural" in output_lower
            or "learning" in output_lower
        )
        assert has_relevant_content, (
            f"Output should contain relevant content, got: {result.output[:200]}"
        )
        assert len(result.sources) > 0

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
@pytest.mark.local_embeddings
class TestRAGSearchHandlerIntegration:
    """Integration tests for RAG in SearchHandler (using HuggingFace)."""

    @pytest.mark.asyncio
    async def test_search_handler_with_rag(self):
        """SearchHandler should work with RAG tool included."""
        # HuggingFace works without API key for public models
        # Setup: Create RAG service and ingest some evidence
        rag_service = get_rag_service(
            collection_name="test_search_handler",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        evidence_list = [
            Evidence(
                content="Test evidence for search handler integration.",
                citation=Citation(
                    source="pubmed",
                    title="Test Evidence",
                    url="https://example.com/test",
                    date="2024",
                    authors=["Tester"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Create RAG tool with the same service instance to ensure same collection
        rag_tool = create_rag_tool(rag_service=rag_service)

        # Create SearchHandler with the custom RAG tool
        handler = SearchHandler(
            tools=[rag_tool],  # Use our RAG tool with the test's collection
            include_rag=False,  # Don't add another RAG tool (we already added it)
            auto_ingest_to_rag=False,  # Don't auto-ingest (already has data)
        )

        # Execute search
        result = await handler.execute("test evidence", max_results_per_tool=5)

        # Assert
        assert result.total_found > 0
        assert "rag" in result.sources_searched
        assert any(e.citation.source == "rag" for e in result.evidence)

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_search_handler_auto_ingest(self):
        """SearchHandler should auto-ingest evidence into RAG."""
        # HuggingFace works without API key for public models
        # Create empty RAG service
        rag_service = get_rag_service(
            collection_name="test_auto_ingest",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        rag_service.clear_collection()

        # Create mock tool that returns evidence
        from unittest.mock import AsyncMock

        mock_tool = AsyncMock()
        mock_tool.name = "pubmed"
        mock_tool.search = AsyncMock(
            return_value=[
                Evidence(
                    content="Evidence to be ingested",
                    citation=Citation(
                        source="pubmed",
                        title="Test",
                        url="https://example.com",
                        date="2024",
                        authors=[],
                    ),
                )
            ]
        )

        # Create handler with auto-ingest enabled
        handler = SearchHandler(
            tools=[mock_tool],
            include_rag=False,  # Don't include RAG as search tool
            auto_ingest_to_rag=True,
        )
        handler._rag_service = rag_service  # Inject RAG service

        # Execute search
        await handler.execute("test query")

        # Verify evidence was ingested
        rag_results = rag_service.retrieve("Evidence to be ingested", top_k=1)
        assert len(rag_results) > 0

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
@pytest.mark.local_embeddings
class TestRAGHybridSearchIntegration:
    """Integration tests for hybrid search (RAG + database) using HuggingFace."""

    @pytest.mark.asyncio
    async def test_hybrid_search_rag_and_pubmed(self):
        """SearchHandler should support RAG + PubMed hybrid search."""
        # HuggingFace works without API key for public models
        # Setup: Ingest evidence into RAG
        rag_service = get_rag_service(
            collection_name="test_hybrid",
            use_openai_embeddings=False,
            use_in_memory=True,  # Use in-memory ChromaDB to avoid file system issues
        )
        evidence_list = [
            Evidence(
                content="Previously collected evidence about metformin.",
                citation=Citation(
                    source="pubmed",
                    title="Previous Research",
                    url="https://example.com/prev",
                    date="2024",
                    authors=[],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Note: This test would require real PubMed API access
        # For now, we'll just test that the handler can be created with both tools
        from src.tools.pubmed import PubMedTool

        handler = SearchHandler(
            tools=[PubMedTool()],
            include_rag=True,
            auto_ingest_to_rag=True,
        )

        # Verify handler has both tools
        tool_names = [t.name for t in handler.tools]
        assert "pubmed" in tool_names
        assert "rag" in tool_names

        # Cleanup
        rag_service.clear_collection()
