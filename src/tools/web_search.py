"""Web search tool using DuckDuckGo."""

import asyncio

import structlog
from duckduckgo_search import DDGS

from src.utils.models import Citation, Evidence, SearchResult

logger = structlog.get_logger()


class WebSearchTool:
    """Tool for searching the web using DuckDuckGo."""

    def __init__(self) -> None:
        self._ddgs = DDGS()

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """Execute a web search."""
        try:
            loop = asyncio.get_running_loop()

            def _do_search() -> list[dict[str, str]]:
                # text() returns an iterator, need to list() it or iterate
                return list(self._ddgs.text(query, max_results=max_results))

            raw_results = await loop.run_in_executor(None, _do_search)

            evidence = []
            for r in raw_results:
                ev = Evidence(
                    content=r.get("body", ""),
                    citation=Citation(
                        title=r.get("title", "No Title"),
                        url=r.get("href", ""),
                        source="web",
                        date="Unknown",
                        authors=[],
                    ),
                    relevance=0.0,
                )
                evidence.append(ev)

            return SearchResult(
                query=query, evidence=evidence, sources_searched=["web"], total_found=len(evidence)
            )

        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return SearchResult(
                query=query, evidence=[], sources_searched=["web"], total_found=0, errors=[str(e)]
            )
