"""Gradio UI for DeepCritical agent."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.orchestrator_factory import create_orchestrator
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.tools.websearch import WebTool
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
        tools=[PubMedTool(), WebTool()],
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

    # Create orchestrator (use mock if no API key)
    use_mock = not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    # If magentic mode requested but no keys, fallback/warn
    if mode == "magentic" and use_mock:
        yield (
            "⚠️ **Warning**: Magentic mode requires valid API keys. "
            "Falling back to Mock Simple mode."
        )
        mode = "simple"

    orchestrator = configure_orchestrator(use_mock=use_mock, mode=mode)

    # Run the agent and stream events
    response_parts = []

    try:
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
    Create the Gradio demo interface.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(
        title="DeepCritical - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🧬 DeepCritical
        ## AI-Powered Drug Repurposing Research Agent

        Ask questions about potential drug repurposing opportunities.
        The agent will search PubMed and the web, evaluate evidence, and provide recommendations.

        **Example questions:**
        - "What drugs could be repurposed for Alzheimer's disease?"
        - "Is metformin effective for cancer treatment?"
        - "What existing medications show promise for Long COVID?"
        """)

        gr.ChatInterface(
            fn=research_agent,
            type="messages",
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

        gr.Markdown("""
        ---
        **Note**: This is a research tool and should not be used for medical decisions.
        Always consult healthcare professionals for medical advice.

        Built with 🤖 PydanticAI + 🔬 PubMed + 🦆 DuckDuckGo
        """)

    return demo


def main() -> None:
    """Run the Gradio app."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
