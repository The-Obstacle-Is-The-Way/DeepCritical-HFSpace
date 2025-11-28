"""Gradio UI for DeepCritical agent with MCP server support."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agent_factory.judges import HFInferenceJudgeHandler, JudgeHandler, MockJudgeHandler
from src.orchestrator_factory import create_orchestrator
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.config import settings
from src.utils.models import OrchestratorConfig


def configure_orchestrator(
    use_mock: bool = False,
    mode: str = "simple",
    user_api_key: str | None = None,
    api_provider: str = "openai",
) -> tuple[Any, str]:
    """
    Create an orchestrator instance.

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode ("simple" or "advanced")
        user_api_key: Optional user-provided API key (BYOK)
        api_provider: API provider ("openai" or "anthropic")

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
    elif (
        user_api_key
        or (api_provider == "openai" and os.getenv("OPENAI_API_KEY"))
        or (api_provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
    ):
        model: AnthropicModel | OpenAIModel | None = None
        if user_api_key:
            # Validate key/provider match to prevent silent auth failures
            if api_provider == "openai" and user_api_key.startswith("sk-ant-"):
                raise ValueError("Anthropic key provided but OpenAI provider selected")
            is_openai_key = user_api_key.startswith("sk-") and not user_api_key.startswith(
                "sk-ant-"
            )
            if api_provider == "anthropic" and is_openai_key:
                raise ValueError("OpenAI key provided but Anthropic provider selected")
            if api_provider == "anthropic":
                anthropic_provider = AnthropicProvider(api_key=user_api_key)
                model = AnthropicModel(settings.anthropic_model, provider=anthropic_provider)
            elif api_provider == "openai":
                openai_provider = OpenAIProvider(api_key=user_api_key)
                model = OpenAIModel(settings.openai_model, provider=openai_provider)
            backend_info = f"Paid API ({api_provider.upper()})"
        else:
            backend_info = "Paid API (Env Config)"

        judge_handler = JudgeHandler(model=model)

    # 3. Free Tier (HuggingFace Inference)
    else:
        judge_handler = HFInferenceJudgeHandler()
        backend_info = "Free Tier (Llama 3.1 / Mistral)"

    orchestrator = create_orchestrator(
        search_handler=search_handler,
        judge_handler=judge_handler,
        config=config,
        mode=mode,  # type: ignore
    )

    return orchestrator, backend_info


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
        mode: Orchestrator mode ("simple" or "advanced")
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

    # Check available keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_user_key = bool(user_api_key)
    has_paid_key = has_openai or has_anthropic or has_user_key

    # Advanced mode requires OpenAI specifically (due to agent-framework binding)
    if mode == "advanced" and not (has_openai or (has_user_key and api_provider == "openai")):
        yield (
            "⚠️ **Warning**: Advanced mode currently requires OpenAI API key. "
            "Falling back to simple mode.\n\n"
        )
        mode = "simple"

    # Inform user about their key being used
    if has_user_key:
        yield (
            f"🔑 **Using your {api_provider.upper()} API key** - "
            "Your key is used only for this session and is never stored.\n\n"
        )
    elif not has_paid_key:
        # No paid keys - will use FREE HuggingFace Inference
        yield (
            "🤗 **Free Tier**: Using HuggingFace Inference (Llama 3.1 / Mistral) for AI analysis.\n"
            "For premium models, enter an OpenAI or Anthropic API key below.\n\n"
        )

    # Run the agent and stream events
    response_parts: list[str] = []

    try:
        # use_mock=False - let configure_orchestrator decide based on available keys
        # It will use: Paid API > HF Inference (free tier)
        orchestrator, backend_name = configure_orchestrator(
            use_mock=False,  # Never use mock in production - HF Inference is the free fallback
            mode=mode,
            user_api_key=user_api_key,
            api_provider=api_provider,
        )

        yield f"🧠 **Backend**: {backend_name}\n\n"

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


def create_demo() -> gr.ChatInterface:
    """
    Create the Gradio demo interface with MCP support.

    Returns:
        Configured Gradio Blocks interface with MCP server enabled
    """
    # 1. Unwrapped ChatInterface (Fixes Accordion Bug)
    demo = gr.ChatInterface(
        fn=research_agent,
        title="🧬 DeepCritical",
        description=(
            "*AI-Powered Drug Repurposing Agent — searches PubMed, "
            "ClinicalTrials.gov & Europe PMC*\n\n"
            "---\n"
            "*Research tool only — not for medical advice.*  \n"
            "**MCP Server Active**: Connect Claude Desktop to `/gradio_api/mcp/`"
        ),
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
        ],
        additional_inputs_accordion=gr.Accordion(label="⚙️ Settings", open=False),
        additional_inputs=[
            gr.Radio(
                choices=["simple", "advanced"],
                value="simple",
                label="Orchestrator Mode",
                info=(
                    "Simple: Linear (Free Tier Friendly) | Advanced: Multi-Agent (Requires OpenAI)"
                ),
            ),
            gr.Textbox(
                label="🔑 API Key (Optional - BYOK)",
                placeholder="sk-... or sk-ant-...",
                type="password",
                info="Enter your own API key. Never stored.",
            ),
            gr.Radio(
                choices=["openai", "anthropic"],
                value="openai",
                label="API Provider",
                info="Select the provider for your API key",
            ),
        ],
    )

    return demo


def main() -> None:
    """Run the Gradio app with MCP server enabled."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True,
        ssr_mode=False,  # Fix for intermittent loading/hydration issues in HF Spaces
    )


if __name__ == "__main__":
    main()
