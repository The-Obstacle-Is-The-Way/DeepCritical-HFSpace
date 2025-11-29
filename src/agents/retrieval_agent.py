"""Retrieval agent for web search and context management."""

import structlog
from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient

from src.state import get_magentic_state
from src.tools.web_search import WebSearchTool
from src.utils.config import settings

logger = structlog.get_logger()

_web_search = WebSearchTool()


@ai_function  # type: ignore[arg-type, misc]
async def search_web(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: Search keywords.
        max_results: Maximum results to return (default 10).

    Returns:
        Formatted search results.
    """
    logger.info("Web search starting", query=query, max_results=max_results)
    state = get_magentic_state()

    results = await _web_search.search(query, max_results)
    if not results.evidence:
        logger.info("Web search returned no results", query=query)
        return f"No web results found for: {query}"

    # Store evidence with deduplication and embedding (all handled by memory layer)
    new_count = await state.add_evidence(results.evidence)
    logger.info(
        "Web search complete",
        query=query,
        results_found=len(results.evidence),
        new_evidence=new_count,
    )

    output = [f"Found {len(results.evidence)} web results ({new_count} new stored):\n"]
    for i, r in enumerate(results.evidence[:max_results], 1):
        output.append(f"{i}. **{r.citation.title}**")
        output.append(f"   Source: {r.citation.url}")
        output.append(f"   {r.content[:300]}...\n")

    return "\n".join(output)


def create_retrieval_agent(chat_client: OpenAIChatClient | None = None) -> ChatAgent:
    """Create a retrieval agent.

    Args:
        chat_client: Optional custom chat client.

    Returns:
        ChatAgent configured for retrieval.
    """
    client = chat_client or OpenAIChatClient(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
    )

    return ChatAgent(
        name="RetrievalAgent",
        description="Searches the web and manages context/evidence.",
        instructions="""You are a retrieval specialist.
Use `search_web` to find information on the internet.
Your goal is to gather relevant evidence for the research task.
Always summarize what you found.""",
        chat_client=client,
        tools=[search_web],
    )
