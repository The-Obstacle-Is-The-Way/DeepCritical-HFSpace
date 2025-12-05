"""DeepBoner research workflow definition using LangGraph."""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.graph.nodes import (
    judge_node,
    resolve_node,
    search_node,
    supervisor_node,
    synthesize_node,
)
from src.agents.graph.state import ResearchState
from src.services.embedding_protocol import EmbeddingServiceProtocol


def create_research_graph(
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    embedding_service: EmbeddingServiceProtocol | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the research state graph.

    Args:
        llm: The language model for the supervisor node.
        checkpointer: Optional persistence layer.
        embedding_service: Service for evidence storage and retrieval.
    """
    graph = StateGraph(ResearchState)

    # --- Nodes ---
    # Bind the LLM to the supervisor node using partial
    bound_supervisor = partial(supervisor_node, llm=llm) if llm else supervisor_node

    # Bind embedding service to worker nodes
    # We use partial to inject the service dependency while keeping the node signature clean
    bound_search = (
        partial(search_node, embedding_service=embedding_service)
        if embedding_service
        else search_node
    )
    bound_judge = (
        partial(judge_node, embedding_service=embedding_service)
        if embedding_service
        else judge_node
    )
    bound_resolve = (
        partial(resolve_node, embedding_service=embedding_service)
        if embedding_service
        else resolve_node
    )
    bound_synthesize = (
        partial(synthesize_node, embedding_service=embedding_service)
        if embedding_service
        else synthesize_node
    )

    graph.add_node("supervisor", bound_supervisor)
    graph.add_node("search", bound_search)
    graph.add_node("judge", bound_judge)
    graph.add_node("resolve", bound_resolve)
    graph.add_node("synthesize", bound_synthesize)

    # --- Edges ---
    # All worker nodes report back to supervisor
    graph.add_edge("search", "supervisor")
    graph.add_edge("judge", "supervisor")
    graph.add_edge("resolve", "supervisor")

    # Synthesis is the end
    graph.add_edge("synthesize", END)

    # --- Conditional Routing ---
    # Supervisor decides where to go next based on state["next_step"]
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next_step"],
        {
            "search": "search",
            "judge": "judge",
            "resolve": "resolve",
            "synthesize": "synthesize",
            "finish": END,
        },
    )

    # Entry Point
    graph.set_entry_point("supervisor")

    return graph.compile(checkpointer=checkpointer)
