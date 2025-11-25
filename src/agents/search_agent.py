from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)

from src.orchestrator import SearchHandlerProtocol
from src.utils.models import Citation, Evidence, SearchResult

if TYPE_CHECKING:
    from src.services.embeddings import EmbeddingService


class SearchAgent(BaseAgent):  # type: ignore[misc]
    """Wraps SearchHandler as an AgentProtocol for Magentic."""

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        evidence_store: dict[str, list[Evidence]],
        embedding_service: "EmbeddingService | None" = None,
    ) -> None:
        super().__init__(
            name="SearchAgent",
            description="Searches PubMed and web for drug repurposing evidence",
        )
        self._handler = search_handler
        self._evidence_store = evidence_store
        self._embeddings = embedding_service

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Execute search based on the last user message."""
        # Extract query from messages
        query = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, ChatMessage) and msg.role == Role.USER and msg.text:
                    query = msg.text
                    break
                elif isinstance(msg, str):
                    query = msg
                    break
        elif isinstance(messages, str):
            query = messages
        elif isinstance(messages, ChatMessage) and messages.text:
            query = messages.text

        if not query:
            return AgentRunResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="No query provided")],
                response_id="search-no-query",
            )

        # Execute search
        result: SearchResult = await self._handler.execute(query, max_results_per_tool=10)

        # Update shared evidence store
        if self._embeddings:
            # Deduplicate by semantic similarity (async-safe)
            unique_evidence = await self._embeddings.deduplicate(result.evidence)

            # Also search for semantically related evidence (async-safe)
            related = await self._embeddings.search_similar(query, n_results=5)

            # Merge related evidence not already in results
            # We need to reconstruct Evidence objects from stored data
            existing_urls = {e.citation.url for e in unique_evidence}

            # Also check what's already in the global store to avoid re-adding
            # logic here is a bit complex: deduplicate returned unique from *new search*
            # but we also want related from *previous searches*

            related_evidence = []
            for item in related:
                if item["id"] not in existing_urls:
                    # Create Evidence from stored metadata
                    # Check if metadata has required fields
                    meta = item.get("metadata", {})
                    # Fallback if date missing
                    date = meta.get("date") or "n.d."
                    authors = meta.get("authors")  # Might be list or string depending on how stored
                    if isinstance(authors, str):
                        authors = [authors]
                    if not authors:
                        authors = ["Unknown"]

                    ev = Evidence(
                        content=item["content"],
                        citation=Citation(
                            title=meta.get("title", "Untitled"),
                            url=item["id"],
                            source=meta.get("source", "vector_db"),
                            date=date,
                            authors=authors,
                        ),
                        relevance=item.get("distance", 0.0),  # Use distance/similarity as proxy
                    )
                    related_evidence.append(ev)

            # Combine
            final_new_evidence = unique_evidence + related_evidence

            # Add to global store (deduping against global store)
            global_urls = {e.citation.url for e in self._evidence_store["current"]}
            really_new = [e for e in final_new_evidence if e.citation.url not in global_urls]
            self._evidence_store["current"].extend(really_new)

            # Update result for reporting
            total_new = len(really_new)

        else:
            # Fallback to URL-based deduplication
            existing_urls = {e.citation.url for e in self._evidence_store["current"]}
            new_unique = [e for e in result.evidence if e.citation.url not in existing_urls]
            self._evidence_store["current"].extend(new_unique)
            total_new = len(new_unique)

        # Format response
        # Get latest N items from store or just the new ones
        # Let's show what was found in this run + related

        evidence_to_show = (
            (unique_evidence + related_evidence)
            if self._embeddings and "unique_evidence" in locals()
            else result.evidence
        )

        evidence_text = "\n".join(
            [
                f"- [{e.citation.title}]({e.citation.url}): {e.content[:200]}..."
                for e in evidence_to_show[:5]
            ]
        )

        response_text = (
            f"Found {result.total_found} sources ({total_new} new added to context):\n\n"
            f"{evidence_text}"
        )

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"search-{result.total_found}",
            additional_properties={"evidence": [e.model_dump() for e in evidence_to_show]},
        )

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Streaming wrapper for search (search itself isn't streaming)."""
        result = await self.run(messages, thread=thread, **kwargs)
        # Yield single update with full result
        yield AgentRunResponseUpdate(messages=result.messages, response_id=result.response_id)
