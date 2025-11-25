from collections.abc import AsyncIterable
from typing import Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
)

from src.orchestrator import SearchHandlerProtocol
from src.utils.models import Evidence, SearchResult


class SearchAgent(BaseAgent):  # type: ignore[misc]
    """Wraps SearchHandler as an AgentProtocol for Magentic."""

    def __init__(
        self,
        search_handler: SearchHandlerProtocol,
        evidence_store: dict[str, list[Evidence]],
    ) -> None:
        super().__init__(
            name="SearchAgent",
            description="Searches PubMed and web for drug repurposing evidence",
        )
        self._handler = search_handler
        self._evidence_store = evidence_store

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
        # We append new evidence, deduplicating by URL is handled in Orchestrator usually,
        # but here we should probably add to the list.
        # For simplicity in this MVP phase, we just extend the list.
        # Ideally, we should dedupe.
        existing_urls = {e.citation.url for e in self._evidence_store["current"]}
        new_unique = [e for e in result.evidence if e.citation.url not in existing_urls]
        self._evidence_store["current"].extend(new_unique)

        # Format response
        evidence_text = "\n".join(
            [
                f"- [{e.citation.title}]({e.citation.url}): {e.content[:200]}..."
                for e in result.evidence[:5]
            ]
        )

        response_text = (
            f"Found {result.total_found} sources ({len(new_unique)} new):\n\n{evidence_text}"
        )

        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)],
            response_id=f"search-{result.total_found}",
            additional_properties={"evidence": [e.model_dump() for e in result.evidence]},
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
