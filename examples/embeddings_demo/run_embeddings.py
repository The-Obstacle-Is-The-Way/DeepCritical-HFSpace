#!/usr/bin/env python3
"""
Demo: Semantic Search & Deduplication (Phase 6).

This script demonstrates embedding-based capabilities:
- Text embedding with sentence-transformers
- Semantic similarity search via ChromaDB
- Duplicate detection by meaning (not just URL)

Usage:
    uv run python examples/embeddings_demo/run_embeddings.py

No API keys required - uses local sentence-transformers model.
"""

import asyncio

from src.services.embeddings import EmbeddingService
from src.utils.models import Citation, Evidence


def create_sample_evidence() -> list[Evidence]:
    """Create sample evidence with some semantic duplicates."""
    return [
        Evidence(
            content="Metformin activates AMPK which inhibits mTOR signaling pathway.",
            citation=Citation(
                source="pubmed",
                title="Metformin and AMPK activation",
                url="https://pubmed.ncbi.nlm.nih.gov/11111/",
                date="2023",
                authors=["Smith J"],
            ),
        ),
        Evidence(
            content="The drug metformin works by turning on AMPK, blocking the mTOR pathway.",
            citation=Citation(
                source="pubmed",
                title="AMPK-mTOR axis in diabetes treatment",
                url="https://pubmed.ncbi.nlm.nih.gov/22222/",
                date="2022",
                authors=["Jones A"],
            ),
        ),
        Evidence(
            content="Sildenafil increases nitric oxide signaling for vasodilation.",
            citation=Citation(
                source="web",
                title="How Viagra Works",
                url="https://example.com/viagra-mechanism",
                date="2023",
                authors=["WebMD"],
            ),
        ),
        Evidence(
            content="Clinical trials show metformin reduces cancer incidence in diabetic patients.",
            citation=Citation(
                source="pubmed",
                title="Metformin and cancer prevention",
                url="https://pubmed.ncbi.nlm.nih.gov/33333/",
                date="2024",
                authors=["Lee K", "Park S"],
            ),
        ),
        Evidence(
            content="Metformin inhibits mTOR through AMPK activation mechanism.",
            citation=Citation(
                source="pubmed",
                title="mTOR inhibition by Metformin",
                url="https://pubmed.ncbi.nlm.nih.gov/44444/",
                date="2023",
                authors=["Brown M"],
            ),
        ),
    ]


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


async def demo_embedding() -> None:
    """Demo single text embedding."""
    print("\n" + "=" * 60)
    print("1. TEXT EMBEDDING DEMO")
    print("=" * 60)

    service = create_fresh_service("embed")

    texts = [
        "Metformin activates AMPK",
        "Aspirin reduces inflammation",
        "Metformin turns on the AMPK enzyme",
    ]

    print("\nEmbedding sample texts...")
    embeddings = await service.embed_batch(texts)

    for text, emb in zip(texts, embeddings, strict=False):
        print(f"  '{text[:40]}...' -> [{emb[0]:.4f}, {emb[1]:.4f}, ... ] (dim={len(emb)})")

    # Calculate similarity between text 0 and text 2 (semantically similar)
    import numpy as np

    sim_0_2 = np.dot(embeddings[0], embeddings[2]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
    )
    sim_0_1 = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )

    print(f"\nSimilarity (Metformin AMPK) vs (Metformin turns on AMPK): {sim_0_2:.3f}")
    print(f"Similarity (Metformin AMPK) vs (Aspirin inflammation):    {sim_0_1:.3f}")
    print("  -> Semantically similar texts have higher cosine similarity!")


async def demo_semantic_search() -> None:
    """Demo semantic similarity search."""
    print("\n" + "=" * 60)
    print("2. SEMANTIC SEARCH DEMO")
    print("=" * 60)

    service = create_fresh_service("search")

    # Add some documents to the vector store
    docs = [
        ("doc1", "Metformin activates AMPK enzyme in liver cells", {"source": "pubmed"}),
        ("doc2", "Aspirin inhibits COX-2 to reduce inflammation", {"source": "pubmed"}),
        ("doc3", "Statins lower cholesterol by inhibiting HMG-CoA reductase", {"source": "web"}),
        ("doc4", "AMPK activation leads to improved glucose metabolism", {"source": "pubmed"}),
        ("doc5", "Sildenafil works via nitric oxide pathway", {"source": "web"}),
    ]

    print("\nIndexing documents...")
    for doc_id, content, meta in docs:
        await service.add_evidence(doc_id, content, meta)
        print(f"  Added: {doc_id}")

    # Search for semantically related content
    query = "drugs that activate AMPK"
    print(f"\nSearching for: '{query}'")

    results = await service.search_similar(query, n_results=3)

    print("\nTop 3 results:")
    for i, r in enumerate(results, 1):
        # Lower distance = more similar (cosine distance: 0=identical, 2=opposite)
        similarity = 1 - r["distance"]
        print(f"  {i}. [{similarity:.2%} similar] {r['content'][:60]}...")


async def demo_deduplication() -> None:
    """Demo semantic deduplication."""
    print("\n" + "=" * 60)
    print("3. SEMANTIC DEDUPLICATION DEMO")
    print("=" * 60)

    # Create fresh service for clean demo
    service = create_fresh_service("dedup")

    evidence = create_sample_evidence()
    print(f"\nOriginal evidence count: {len(evidence)}")
    for i, e in enumerate(evidence, 1):
        print(f"  {i}. {e.citation.title}")

    print("\nRunning semantic deduplication (threshold=0.85)...")
    unique = await service.deduplicate(evidence, threshold=0.85)

    print(f"\nUnique evidence count: {len(unique)}")
    print(f"Removed {len(evidence) - len(unique)} semantic duplicates\n")

    for i, e in enumerate(unique, 1):
        print(f"  {i}. {e.citation.title}")

    print("\n  -> Notice: Papers about 'Metformin AMPK mTOR' were deduplicated!")
    print("     Different titles, same semantic meaning = duplicate removed.")


async def main() -> None:
    """Run all embedding demos."""
    print("\n" + "=" * 60)
    print("DeepCritical Embeddings Demo (Phase 6)")
    print("Using: sentence-transformers + ChromaDB")
    print("=" * 60)

    await demo_embedding()
    await demo_semantic_search()
    await demo_deduplication()

    print("\n" + "=" * 60)
    print("Demo complete! Embeddings enable:")
    print("  - Finding papers by MEANING, not just keywords")
    print("  - Removing duplicate findings automatically")
    print("  - Building diverse evidence sets for research")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
