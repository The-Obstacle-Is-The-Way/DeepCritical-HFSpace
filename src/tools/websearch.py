"""Web search tool using DuckDuckGo."""

import asyncio
from typing import Any

from duckduckgo_search import DDGS

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class WebTool:
    """Search tool for general web search via DuckDuckGo."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "web"

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search DuckDuckGo and return evidence.

        Note: duckduckgo-search is synchronous, so we run it in executor.
        """
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_search(query, max_results),
            )
            return results
        except Exception as e:
            raise SearchError(f"Web search failed: {e}") from e

    def _sync_search(self, query: str, max_results: int) -> list[Evidence]:
        """Synchronous search implementation."""
        evidence_list = []

        with DDGS() as ddgs:
            results: list[dict[str, Any]] = list(ddgs.text(query, max_results=max_results))

        for result in results:
            evidence_list.append(
                Evidence(
                    content=result.get("body", "")[:1000],
                    citation=Citation(
                        source="web",
                        title=result.get("title", "Unknown")[:500],
                        url=result.get("href", ""),
                        date="Unknown",
                        authors=[],
                    ),
                )
            )

        return evidence_list
