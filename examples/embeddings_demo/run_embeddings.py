#!/usr/bin/env python3
"""
Demo: Semantic Search & Deduplication (Phase 6).

This script demonstrates embedding-based capabilities using REAL data:
- Fetches REAL abstracts from PubMed
- Embeds text with sentence-transformers
- Performs semantic deduplication on LIVE research data

Usage:
    uv run python examples/embeddings_demo/run_embeddings.py
"""

import asyncio

from src.services.embeddings import EmbeddingService
from src.tools.pubmed import PubMedTool


def create_fresh_service(name_suffix: str = "") -> EmbeddingService:
    """Create a fresh embedding service with unique collection name."""
    import uuid

    # Create service with unique collection by modifying the internal collection
    service = EmbeddingService.__new__(EmbeddingService)
    service._model = __import__("sentence_transformers").SentenceTransformer("all-MiniLM-L6-v2")
    service._client = __import__("chromadb").Client()
    collection_name = f"demo_{name_suffix}_{uuid.uuid4().hex[:8]}"
    service._collection = service._client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    return service


async def demo_real_pipeline() -> None:
    """Run the demo using REAL PubMed data."""
    print("\n" + "=" * 60)
    print("DeepBoner Embeddings Demo (REAL DATA)")
    print("=" * 60)

    # 1. Fetch Real Data
    query = "metformin mechanism of action"
    print(f"\n[1] Fetching real papers for: '{query}'...")
    pubmed = PubMedTool()
    # Fetch enough results to likely get some overlap/redundancy
    evidence = await pubmed.search(query, max_results=10)

    print(f"    Found {len(evidence)} papers.")
    print("\n    Sample Titles:")
    for i, e in enumerate(evidence[:3], 1):
        print(f"    {i}. {e.citation.title[:80]}...")

    # 2. Embed Data
    print("\n[2] Embedding abstracts (sentence-transformers)...")
    service = create_fresh_service("real_demo")

    # 3. Semantic Search
    print("\n[3] Semantic Search Demo")
    print("    Indexing evidence...")
    for e in evidence:
        # Use URL as ID for uniqueness
        await service.add_evidence(
            evidence_id=e.citation.url,
            content=e.content,
            metadata={
                "source": e.citation.source,
                "title": e.citation.title,
                "date": e.citation.date,
            },
        )

    semantic_query = "activation of AMPK pathway"
    print(f"    Searching for concept: '{semantic_query}'")
    results = await service.search_similar(semantic_query, n_results=2)

    print("    Top matches:")
    for i, r in enumerate(results, 1):
        similarity = 1 - r["distance"]
        print(f"    {i}. [{similarity:.1%} match] {r['metadata']['title'][:70]}...")

    # 4. Semantic Deduplication
    print("\n[4] Semantic Deduplication Demo")
    # Create a FRESH service for deduplication so we don't clash with Step 3's index
    dedup_service = create_fresh_service("dedup_demo")

    print("    Checking for redundant papers (threshold=0.85)...")

    # To force a duplicate for demo purposes, let's double the evidence list
    # simulating finding the same papers again or very similar ones
    duplicated_evidence = evidence + evidence[:2]
    print(f"    Input pool: {len(duplicated_evidence)} items (with artificial duplicates added)")

    unique = await dedup_service.deduplicate(duplicated_evidence, threshold=0.85)

    print(f"    Output pool: {len(unique)} unique items")
    print(f"    Removed {len(duplicated_evidence) - len(unique)} duplicates.")

    print("\n" + "=" * 60)
    print("Demo complete! Verified with REAL PubMed data.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_real_pipeline())
