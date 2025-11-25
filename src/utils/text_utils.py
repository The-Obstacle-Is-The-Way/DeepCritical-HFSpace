"""Text processing utilities for evidence handling."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService
    from src.utils.models import Evidence


def truncate_at_sentence(text: str, max_chars: int = 300) -> str:
    """Truncate text at sentence boundary, preserving meaning.

    Args:
        text: The text to truncate
        max_chars: Maximum characters (default 300)

    Returns:
        Text truncated at last complete sentence within limit
    """
    if len(text) <= max_chars:
        return text

    # Find truncation point
    truncated = text[:max_chars]

    # Look for sentence endings: . ! ? followed by space or end
    # We check for sep at the END of the truncated string
    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars // 2:  # Don't truncate too aggressively (less than half)
            return text[: last_sep + 1].strip()

    # Fallback: find last period (even if not followed by space, e.g. end of string)
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return text[: last_period + 1].strip()

    # Last resort: truncate at word boundary
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return text[:last_space].strip() + "..."

    return truncated + "..."


async def select_diverse_evidence(
    evidence: list["Evidence"], n: int, query: str, embeddings: "EmbeddingService | None" = None
) -> list["Evidence"]:
    """Select n most diverse and relevant evidence items.

    Uses Maximal Marginal Relevance (MMR) when embeddings available,
    falls back to relevance_score sorting otherwise.

    Args:
        evidence: All available evidence
        n: Number of items to select
        query: Original query for relevance scoring
        embeddings: Optional EmbeddingService for semantic diversity

    Returns:
        Selected evidence items, diverse and relevant
    """
    if not evidence:
        return []

    if n >= len(evidence):
        return evidence

    # Fallback: sort by relevance score if no embeddings
    if embeddings is None:
        return sorted(
            evidence,
            key=lambda e: e.relevance,  # Use .relevance (from Pydantic model)
            reverse=True,
        )[:n]

    # MMR: Maximal Marginal Relevance for diverse selection
    # Score = λ * relevance - (1-λ) * max_similarity_to_selected
    lambda_param = 0.7  # Balance relevance vs diversity

    # Get query embedding
    query_emb = await embeddings.embed(query)

    # Get all evidence embeddings
    evidence_embs = await embeddings.embed_batch([e.content for e in evidence])

    # Cosine similarity helper
    def cosine(a: list[float], b: list[float]) -> float:
        arr_a, arr_b = np.array(a), np.array(b)
        denominator = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
        if denominator == 0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / denominator)

    # Compute relevance scores (cosine similarity to query)
    # Note: We use semantic relevance to query, not the keyword search 'relevance' score
    relevance_scores = [cosine(query_emb, emb) for emb in evidence_embs]

    # Greedy MMR selection
    selected_indices: list[int] = []
    remaining = set(range(len(evidence)))

    for _ in range(n):
        best_score = float("-inf")
        best_idx = -1

        for idx in remaining:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component: max similarity to already selected
            if selected_indices:
                max_sim = max(
                    cosine(evidence_embs[idx], evidence_embs[sel]) for sel in selected_indices
                )
            else:
                max_sim = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [evidence[i] for i in selected_indices]
