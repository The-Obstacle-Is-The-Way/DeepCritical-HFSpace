"""LangGraph-based orchestrator implementation.

NOTE: This orchestrator is deprecated in favor of the shared memory layer
integrated into Simple and Advanced modes (SPEC-08). It remains as a reference
implementation for LangGraph patterns.
"""

import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agents.graph.state import ResearchState
from src.agents.graph.workflow import create_research_graph
from src.orchestrators.base import OrchestratorProtocol
from src.utils.config import settings
from src.utils.models import AgentEvent
from src.utils.service_loader import get_embedding_service


class LangGraphOrchestrator(OrchestratorProtocol):
    """State-driven research orchestrator using LangGraph.

    DEPRECATED: Memory features are now integrated into Simple and Advanced modes.
    This class is kept for reference and potential future use.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        checkpoint_path: str | None = None,
        api_key: str | None = None,
    ):
        self._max_iterations = max_iterations
        self._checkpoint_path = checkpoint_path
        self._api_key = api_key

        # Initialize the LLM (Qwen 2.5 via HF Inference)
        # We use the serverless API by default
        # FIX: Use 7B model to stay on HuggingFace native infrastructure
        # Large models (70B+) route to Novita/Hyperbolic providers (500/401 errors)
        repo_id = settings.huggingface_model or "Qwen/Qwen2.5-7B-Instruct"

        # Determine HF Token (BYOK > Env)
        # Note: If api_key starts with 'sk-', it's likely OpenAI, which isn't supported here
        # for the LLM, but we store it for the embedding service.
        hf_token = settings.hf_token
        if api_key and not api_key.startswith("sk-"):
            hf_token = api_key

        if not hf_token:
            # If we have an OpenAI key but no HF token, we can't run the HF LLM
            if api_key and api_key.startswith("sk-"):
                raise ValueError(
                    "LangGraphOrchestrator currently requires a Hugging Face token (HF_TOKEN) "
                    "for the LLM, even if using OpenAI for embeddings. "
                    "Please use Advanced Mode for OpenAI support."
                )
            raise ValueError(
                "HF_TOKEN (Hugging Face API Token) is required for LangGraph orchestrator."
            )

        self.llm_endpoint = HuggingFaceEndpoint(  # type: ignore
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.1,
            huggingfacehub_api_token=hf_token,
        )
        self.chat_model = ChatHuggingFace(llm=self.llm_endpoint)

    async def run(self, query: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute research workflow with structured state."""
        # Initialize embedding service using tiered selection (service_loader)
        # Returns LlamaIndexRAGService if OpenAI key available, else local EmbeddingService
        embedding_service = get_embedding_service(api_key=self._api_key)

        # Setup checkpointer (SQLite for dev)
        if self._checkpoint_path:
            # Ensure directory exists (handle paths without directory component)
            dir_name = os.path.dirname(self._checkpoint_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            saver = AsyncSqliteSaver.from_conn_string(self._checkpoint_path)
        else:
            saver = None

        # Use a helper context manager to handle the optional saver
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def get_graph_context(saver_instance: Any) -> AsyncIterator[Any]:
            if saver_instance:
                async with saver_instance as s:
                    yield create_research_graph(
                        llm=self.chat_model,
                        checkpointer=s,
                        embedding_service=embedding_service,
                    )
            else:
                yield create_research_graph(
                    llm=self.chat_model,
                    checkpointer=None,
                    embedding_service=embedding_service,
                )

        async with get_graph_context(saver) as graph:
            # Initialize state
            initial_state: ResearchState = {
                "query": query,
                "hypotheses": [],
                "conflicts": [],
                "evidence_ids": [],
                "messages": [],
                "next_step": "search",  # Start with search
                "iteration_count": 0,
                "max_iterations": self._max_iterations,
            }

            yield AgentEvent(type="started", message=f"Starting LangGraph research: {query}")

            # Config for persistence (unique thread_id per run to avoid state conflicts)
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}} if saver else {}

            # Stream events
            # We use astream to get updates from the graph
            async for event in graph.astream(initial_state, config=config):
                # Event is a dict of node_name -> state_update
                for node_name, update in event.items():
                    if update.get("messages"):
                        last_msg = update["messages"][-1]
                        event_type: Literal["progress", "thinking", "searching"] = "progress"
                        if node_name == "supervisor":
                            event_type = "thinking"
                        elif node_name == "search":
                            event_type = "searching"

                        yield AgentEvent(
                            type=event_type, message=str(last_msg.content), data={"node": node_name}
                        )
                    elif node_name == "supervisor":
                        yield AgentEvent(
                            type="thinking",
                            message=f"Supervisor decided: {update.get('next_step')}",
                            data={"node": node_name},
                        )

            yield AgentEvent(type="complete", message="Research complete.")
