"""Gradio UI for DeepCritical agent with MCP server support."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.mcp_tools import (
    analyze_hypothesis,
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
from src.utils.config import settings
from src.utils.models import OrchestratorConfig


def configure_orchestrator(
    use_mock: bool = False,
    mode: str = "simple",
    user_api_key: str | None = None,
    api_provider: str = "openai",
) -> Any:
    """
    Create an orchestrator instance.

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode ("simple" or "magentic")
        user_api_key: Optional user-provided API key (BYOK)
        api_provider: API provider ("openai" or "anthropic")

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
        # Create model with user's API key if provided
        model: AnthropicModel | OpenAIModel | None = None
        if user_api_key:
            if api_provider == "anthropic":
                anthropic_provider = AnthropicProvider(api_key=user_api_key)
                model = AnthropicModel(settings.anthropic_model, provider=anthropic_provider)
            elif api_provider == "openai":
                openai_provider = OpenAIProvider(api_key=user_api_key)
                model = OpenAIModel(settings.openai_model, provider=openai_provider)
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
        judge_handler = JudgeHandler(model=model)

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
    api_key: str = "",
    api_provider: str = "openai",
) -> AsyncGenerator[str, None]:
    """
    Gradio chat function that runs the research agent.

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        mode: Orchestrator mode ("simple" or "magentic")
        api_key: Optional user-provided API key (BYOK - Bring Your Own Key)
        api_provider: API provider ("openai" or "anthropic")

    Yields:
        Markdown-formatted responses for streaming
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    # Clean user-provided API key
    user_api_key = api_key.strip() if api_key else None

    # Decide whether to use real LLMs or mock based on mode and available keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_user_key = bool(user_api_key)

    if mode == "magentic":
        # Magentic currently supports OpenAI only
        use_mock = not (has_openai or (has_user_key and api_provider == "openai"))
    else:
        # Simple mode can work with either provider
        use_mock = not (has_openai or has_anthropic or has_user_key)

    # If magentic mode requested but no OpenAI key, fallback/warn
    if mode == "magentic" and use_mock:
        yield (
            "⚠️ **Warning**: Magentic mode requires OpenAI API key. "
            "Falling back to demo mode.\n\n"
        )
        mode = "simple"

    # Inform user about their key being used
    if has_user_key and not use_mock:
        yield (
            f"🔑 **Using your {api_provider.upper()} API key** - "
            "Your key is used only for this session and is never stored.\n\n"
        )

    # Warn users when running in demo mode (no LLM keys)
    if use_mock:
        yield (
            "🔬 **Demo Mode**: Running with real biomedical searches but without "
            "LLM-powered analysis.\n\n"
            "**To unlock full AI analysis:**\n"
            "- Enter your OpenAI or Anthropic API key below, OR\n"
            "- Configure secrets in HuggingFace Space settings\n\n"
            "---\n\n"
        )

    # Run the agent and stream events
    response_parts: list[str] = []

    try:
        orchestrator = configure_orchestrator(
            use_mock=use_mock,
            mode=mode,
            user_api_key=user_api_key,
            api_provider=api_provider,
        )
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
        # fill_height=True removed to prevent header cutoff in HF Spaces
    ) as demo:
        # 1. Title & Description (Top of page)
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

        # 2. Main chat interface
        # Note: additional_inputs will render in an accordion below the chat input by default.
        # This is standard Gradio ChatInterface behavior and ensures a clean layout.
        gr.ChatInterface(
            fn=research_agent,
            examples=[
                [
                    "What drugs could be repurposed for Alzheimer's disease?",
                    "simple",
                    "",
                    "openai",
                ],
                [
                    "Is metformin effective for treating cancer?",
                    "simple",
                    "",
                    "openai",
                ],
                [
                    "What medications show promise for Long COVID treatment?",
                    "simple",
                    "",
                    "openai",
                ],
                [
                    "Can statins be repurposed for neurological conditions?",
                    "simple",
                    "",
                    "openai",
                ],
            ],
            additional_inputs=[
                gr.Radio(
                    choices=["simple", "magentic"],
                    value="simple",
                    label="Orchestrator Mode",
                    info="Simple: Linear (OpenAI/Anthropic) | Magentic: Multi-Agent (OpenAI)",
                ),
                gr.Textbox(
                    label="🔑 API Key (Optional - Bring Your Own Key)",
                    placeholder="sk-... or sk-ant-...",
                    type="password",
                    info="Enter your own API key for full AI analysis. Never stored.",
                ),
                gr.Radio(
                    choices=["openai", "anthropic"],
                    value="openai",
                    label="API Provider",
                    info="Select the provider for your API key",
                ),
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

        with gr.Tab("Analyze Hypothesis"):
            gr.Interface(
                fn=analyze_hypothesis,
                inputs=[
                    gr.Textbox(label="Drug", placeholder="metformin"),
                    gr.Textbox(label="Condition", placeholder="Alzheimer's disease"),
                    gr.Textbox(
                        label="Evidence Summary",
                        placeholder="Studies show metformin reduces tau phosphorylation...",
                        lines=5,
                    ),
                ],
                outputs=gr.Markdown(label="Analysis Result"),
                api_name="analyze_hypothesis",
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
    # CSS to fix the header cutoff issue in HuggingFace Spaces
    # Adds padding to the top of the container to clear the HF banner
    css = """
    .gradio-container {
        padding-top: 50px !important;
    }
    """

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True,
        css=css,
    )


if __name__ == "__main__":
    main()
