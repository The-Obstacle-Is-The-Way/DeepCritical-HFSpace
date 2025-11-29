# P3: Ephemeral Memory Architecture (No Persistence)

**Status:** OPEN
**Priority:** P3 (Feature/Architecture Gap)
**Found By:** Codebase Investigation
**Date:** 2025-11-29

## Description
The current `EmbeddingService` (`src/services/embeddings.py`) initializes an **in-memory** ChromaDB client (`chromadb.Client()`) and creates a random UUID-based collection for every new session.

While `src/utils/config.py` defines a `chroma_db_path` for persistence, it is currently **ignored**.

## Impact
1.  **No Long-Term Learning:** The agent cannot "remember" research from previous runs. Every time you restart the app, it starts from zero.
2.  **Redundant Costs:** If a user researches "Diabetes" twice, the agent re-searches and re-embeds the same papers, wasting tokens and compute time.

## Technical Details
- **Current:** `self._client = chromadb.Client()` (In-Memory)
- **Required:** `self._client = chromadb.PersistentClient(path=settings.chroma_db_path)`

## Recommendation
For a "Hackathon Demo," this is **low priority** (ephemeral is fine).
For a "Real Product," this is **critical** (users expect a library of research).
