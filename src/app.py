"""Gradio UI for DeepBoner agent with MCP server support."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agent_factory.judges import HFInferenceJudgeHandler, JudgeHandler, MockJudgeHandler
from src.orchestrator_factory import create_orchestrator
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError
from src.utils.models import OrchestratorConfig


def configure_orchestrator(
    use_mock: bool = False,
    mode: str = "simple",
    user_api_key: str | None = None,
) -> tuple[Any, str]:
    """
    Create an orchestrator instance.

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode ("simple" or "advanced")
        user_api_key: Optional user-provided API key (BYOK) - auto-detects provider

    Returns:
        Tuple of (Orchestrator instance, backend_name)
    """
    # Create orchestrator config
    config = OrchestratorConfig(
        max_iterations=10,
        max_results_per_tool=10,
    )

    # Create search tools
    search_handler = SearchHandler(
        tools=[PubMedTool(), ClinicalTrialsTool(), EuropePMCTool()],
        timeout=config.search_timeout,
    )

    # Create judge (mock, real, or free tier)
    judge_handler: JudgeHandler | MockJudgeHandler | HFInferenceJudgeHandler
    backend_info = "Unknown"

    # 1. Forced Mock (Unit Testing)
    if use_mock:
        judge_handler = MockJudgeHandler()
        backend_info = "Mock (Testing)"

    # 2. Paid API Key (User provided or Env)
    elif user_api_key and user_api_key.strip():
        # Auto-detect provider from key prefix
        model: AnthropicModel | OpenAIChatModel
        if user_api_key.startswith("sk-ant-"):
            # Anthropic key
            anthropic_provider = AnthropicProvider(api_key=user_api_key)
            model = AnthropicModel(settings.anthropic_model, provider=anthropic_provider)
            backend_info = "Paid API (Anthropic)"
        elif user_api_key.startswith("sk-"):
            # OpenAI key
            openai_provider = OpenAIProvider(api_key=user_api_key)
            model = OpenAIChatModel(settings.openai_model, provider=openai_provider)
            backend_info = "Paid API (OpenAI)"
        else:
            raise ConfigurationError(
                "Invalid API key format. Expected sk-... (OpenAI) or sk-ant-... (Anthropic)"
            )
        judge_handler = JudgeHandler(model=model)

    # 3. Environment API Keys (fallback)
    elif os.getenv("OPENAI_API_KEY"):
        judge_handler = JudgeHandler(model=None)  # Uses env key
        backend_info = "Paid API (OpenAI from env)"

    elif os.getenv("ANTHROPIC_API_KEY"):
        judge_handler = JudgeHandler(model=None)  # Uses env key
        backend_info = "Paid API (Anthropic from env)"

    # 4. Free Tier (HuggingFace Inference)
    else:
        judge_handler = HFInferenceJudgeHandler()
        backend_info = "Free Tier (Llama 3.1 / Mistral)"

    orchestrator = create_orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
        mode=mode,  # type: ignore
        api_key=user_api_key,
    )

    return orchestrator, backend_info


async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    mode: str = "simple",
    api_key: str = "",
    api_key_state: str = "",
) -> AsyncGenerator[str, None]:
    """
    Gradio chat function that runs the research agent.

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        mode: Orchestrator mode ("simple" or "advanced")
        api_key: Optional user-provided API key (BYOK - auto-detects provider)
        api_key_state: Persistent API key state (survives example clicks)

    Yields:
        Markdown-formatted responses for streaming
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    # BUG FIX: Prefer freshly-entered key, then persisted state
    user_api_key = (api_key.strip() or api_key_state.strip()) or None

    # Check available keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    # Check for OpenAI user key
    is_openai_user_key = (
        user_api_key and user_api_key.startswith("sk-") and not user_api_key.startswith("sk-ant-")
    )
    has_paid_key = has_openai or has_anthropic or bool(user_api_key)

    # Advanced mode requires OpenAI specifically (due to agent-framework binding)
    if mode == "advanced" and not (has_openai or is_openai_user_key):
        yield (
            "⚠️ **Warning**: Advanced mode currently requires OpenAI API key. "
            "Anthropic keys only work in Simple mode. Falling back to Simple.\n\n"
        )
        mode = "simple"

    # Inform user about fallback if no keys
    if not has_paid_key:
        # No paid keys - will use FREE HuggingFace Inference
        yield (
            "🤗 **Free Tier**: Using HuggingFace Inference (Llama 3.1 / Mistral) for AI analysis.\n"
            "For premium models, enter an OpenAI or Anthropic API key below.\n\n"
        )

    # Run the agent and stream events
    response_parts: list[str] = []
    streaming_buffer = ""  # Buffer for accumulating streaming tokens

    try:
        # use_mock=False - let configure_orchestrator decide based on available keys
        # It will use: Paid API > HF Inference (free tier)
        orchestrator, backend_name = configure_orchestrator(
            use_mock=False,  # Never use mock in production - HF Inference is the free fallback
            mode=mode,
            user_api_key=user_api_key,
        )

        yield f"🧠 **Backend**: {backend_name}\n\n"

        async for event in orchestrator.run(message):
            # BUG FIX: Handle streaming events separately to avoid token-by-token spam
            if event.type == "streaming":
                # Accumulate streaming tokens without emitting individual events
                streaming_buffer += event.message
                # Yield the current buffer combined with previous parts to show progress
                # But DO NOT append to response_parts list yet (to avoid O(N^2) list growth)
                current_parts = [*response_parts, f"📡 **STREAMING**: {streaming_buffer}"]
                yield "\n\n".join(current_parts)
                continue

            # For non-streaming events, flush any buffered streaming content first
            if streaming_buffer:
                response_parts.append(f"📡 **STREAMING**: {streaming_buffer}")
                streaming_buffer = ""  # Reset buffer

            # Handle complete events specially
            if event.type == "complete":
                yield event.message
            else:
                # Format and append non-streaming events
                event_md = event.to_markdown()
                response_parts.append(event_md)
                # Show progress
                yield "\n\n".join(response_parts)

        # Flush any remaining streaming content at the end
        if streaming_buffer:
            response_parts.append(f"📡 **STREAMING**: {streaming_buffer}")
            yield "\n\n".join(response_parts)

    except Exception as e:
        yield f"❌ **Error**: {e!s}"


def create_demo() -> tuple[gr.ChatInterface, gr.Accordion]:
    """
    Create the Gradio demo interface with MCP support.

    Returns:
        Configured Gradio Blocks interface with MCP server enabled
    """
    additional_inputs_accordion = gr.Accordion(
        label="⚙️ Mode & API Key (Free tier works!)", open=False
    )

    # BUG FIX: Add gr.State for API key persistence across example clicks
    api_key_state = gr.State("")

    # 1. Unwrapped ChatInterface (Fixes Accordion Bug)
    demo = gr.ChatInterface(
        fn=research_agent,
        title="🍆 DeepBoner",
        description=(
            "*AI-Powered Sexual Health Research Agent — searches PubMed, "
            "ClinicalTrials.gov & Europe PMC*\n\n"
            "Deep research for sexual wellness, ED treatments, hormone therapy, "
            "libido, and reproductive health - for all genders.\n\n"
            "---\n"
            "*Research tool only — not for medical advice.*  \n"
            "**MCP Server Active**: Connect Claude Desktop to `/gradio_api/mcp/`"
        ),
        examples=[
            [
                "What drugs improve female libido post-menopause?",
                "simple",
                # Removed empty strings for api_key and api_key_state to prevent overwriting
            ],
            [
                "Clinical trials for erectile dysfunction alternatives to PDE5 inhibitors?",
                "advanced",
            ],
            [
                "Evidence for testosterone therapy in women with HSDD?",
                "simple",
            ],
        ],
        additional_inputs_accordion=additional_inputs_accordion,
        additional_inputs=[
            gr.Radio(
                choices=["simple", "advanced"],
                value="simple",
                label="Orchestrator Mode",
                info="⚡ Simple: Free/OpenAI/Anthropic | 🔬 Advanced: OpenAI only",
            ),
            gr.Textbox(
                label="🔑 API Key (Optional)",
                placeholder="sk-... (OpenAI) or sk-ant-... (Anthropic)",
                type="password",
                info="Leave empty for free tier. Auto-detects provider from key prefix.",
            ),
            api_key_state,  # Hidden state component for persistence
        ],
    )

    # API key persists because examples only include [message, mode] columns,
    # so Gradio doesn't overwrite the api_key textbox when examples are clicked.

    return demo, additional_inputs_accordion


def main() -> None:
    """Run the Gradio app with MCP server enabled."""
    demo, _ = create_demo()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),  # nosec B104
        server_port=7860,
        share=False,
        mcp_server=True,
        ssr_mode=False,  # Fix for intermittent loading/hydration issues in HF Spaces
    )


if __name__ == "__main__":
    main()
