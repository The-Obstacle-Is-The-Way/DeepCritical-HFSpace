"""Gradio UI for DeepCritical agent with MCP server support."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.mcp_tools import (
    search_all_sources,
    search_biorxiv,
    search_clinical_trials,
    search_pubmed,
)
from src.orchestrator_factory import create_orchestrator
from src.tools.biorxiv import BioRxivTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.models import OrchestratorConfig


def configure_orchestrator(use_mock: bool = False, mode: str = "simple") -> Any:
    """
    Create an orchestrator instance.

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode ("simple" or "magentic")

    Returns:
        Configured Orchestrator instance
    """
    # Create orchestrator config
    config = OrchestratorConfig(
        max_iterations=5,
        max_results_per_tool=10,
    )

    # Create search tools
    search_handler = SearchHandler(
        tools=[PubMedTool(), ClinicalTrialsTool(), BioRxivTool()],
        timeout=config.search_timeout,
    )

    # Create judge (mock or real)
    judge_handler: JudgeHandler | MockJudgeHandler
    if use_mock:
        judge_handler = MockJudgeHandler()
    else:
        judge_handler = JudgeHandler()

    return create_orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
        mode=mode,  # type: ignore
    )


async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    mode: str = "simple",
) -> AsyncGenerator[str, None]:
    """
    Gradio chat function that runs the research agent.

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        mode: Orchestrator mode ("simple" or "magentic")

    Yields:
        Markdown-formatted responses for streaming
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    # Decide whether to use real LLMs or mock based on mode and available keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if mode == "magentic":
        # Magentic currently supports OpenAI only
        use_mock = not has_openai
    else:
        # Simple mode can work with either provider
        use_mock = not (has_openai or has_anthropic)

    # If magentic mode requested but no OpenAI key, fallback/warn
    if mode == "magentic" and use_mock:
        yield (
            "⚠️ **Warning**: Magentic mode requires OpenAI API key. "
            "Falling back to Mock Simple mode."
        )
        mode = "simple"

    # Run the agent and stream events
    response_parts: list[str] = []

    try:
        orchestrator = configure_orchestrator(use_mock=use_mock, mode=mode)
        async for event in orchestrator.run(message):
            # Format event as markdown
            event_md = event.to_markdown()
            response_parts.append(event_md)

            # If complete, show full response
            if event.type == "complete":
                yield event.message
            else:
                # Show progress
                yield "\n\n".join(response_parts)

    except Exception as e:
        yield f"❌ **Error**: {e!s}"


def create_demo() -> Any:
    """
    Create the Gradio demo interface with MCP support.

    Returns:
        Configured Gradio Blocks interface with MCP server enabled
    """
    with gr.Blocks(
        title="DeepCritical - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🧬 DeepCritical
        ## AI-Powered Drug Repurposing Research Agent

        Ask questions about potential drug repurposing opportunities.
        The agent searches PubMed, ClinicalTrials.gov, and bioRxiv/medRxiv preprints.

        **Example questions:**
        - "What drugs could be repurposed for Alzheimer's disease?"
        - "Is metformin effective for cancer treatment?"
        - "What existing medications show promise for Long COVID?"
        """)

        # Main chat interface (existing)
        gr.ChatInterface(
            fn=research_agent,
            type="messages",  # type: ignore
            title="",
            examples=[
                "What drugs could be repurposed for Alzheimer's disease?",
                "Is metformin effective for treating cancer?",
                "What medications show promise for Long COVID treatment?",
                "Can statins be repurposed for neurological conditions?",
            ],
            additional_inputs=[
                gr.Radio(
                    choices=["simple", "magentic"],
                    value="simple",
                    label="Orchestrator Mode",
                    info="Simple: Linear (OpenAI/Anthropic) | Magentic: Multi-Agent (OpenAI)",
                )
            ],
        )

        # MCP Tool Interfaces (exposed via MCP protocol)
        gr.Markdown("---\n## MCP Tools (Also Available via Claude Desktop)")

        with gr.Tab("PubMed Search"):
            gr.Interface(
                fn=search_pubmed,
                inputs=[
                    gr.Textbox(label="Query", placeholder="metformin alzheimer"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_pubmed",
            )

        with gr.Tab("Clinical Trials"):
            gr.Interface(
                fn=search_clinical_trials,
                inputs=[
                    gr.Textbox(label="Query", placeholder="diabetes phase 3"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_clinical_trials",
            )

        with gr.Tab("Preprints"):
            gr.Interface(
                fn=search_biorxiv,
                inputs=[
                    gr.Textbox(label="Query", placeholder="long covid treatment"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_biorxiv",
            )

        with gr.Tab("Search All"):
            gr.Interface(
                fn=search_all_sources,
                inputs=[
                    gr.Textbox(label="Query", placeholder="metformin cancer"),
                    gr.Slider(1, 20, value=5, step=1, label="Max Per Source"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_all",
            )

        gr.Markdown("""
        ---
        **Note**: This is a research tool and should not be used for medical decisions.
        Always consult healthcare professionals for medical advice.

        Built with PydanticAI + PubMed, ClinicalTrials.gov & bioRxiv

        **MCP Server**: Available at `/gradio_api/mcp/` for Claude Desktop integration
        """)

    return demo


def main() -> None:
    """Run the Gradio app with MCP server enabled."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True,  # Enable MCP server
    )


if __name__ == "__main__":
    main()
