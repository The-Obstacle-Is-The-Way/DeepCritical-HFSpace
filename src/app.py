"Gradio UI for DeepBoner agent with MCP server support."

import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

import gradio as gr

from src.config.domain import ResearchDomain
from src.orchestrators import create_orchestrator
from src.utils.config import settings
from src.utils.exceptions import ConfigurationError
from src.utils.models import OrchestratorConfig
from src.utils.service_loader import warmup_services

OrchestratorMode = Literal["advanced", "hierarchical"]  # Unified Architecture (SPEC-16)


# CSS to force dark mode on API key input
# NOTE: Browser autofill requires -webkit-autofill selectors to override
CUSTOM_CSS = """
.api-key-input input {
    background-color: #1f2937 !important;
    color: white !important;
    border-color: #374151 !important;
}
.api-key-input input:focus,
.api-key-input input:focus-visible {
    background-color: #1f2937 !important;
    color: white !important;
    border-color: #e879f9 !important;
    outline: none !important;
}
/* Override aggressive browser autofill styling */
.api-key-input input:-webkit-autofill,
.api-key-input input:-webkit-autofill:hover,
.api-key-input input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0px 1000px #1f2937 inset !important;
    -webkit-text-fill-color: white !important;
    caret-color: white !important;
    transition: background-color 5000s ease-in-out 0s;
}
"""


def configure_orchestrator(
    use_mock: bool = False,
    mode: OrchestratorMode = "advanced",
    user_api_key: str | None = None,
    domain: str | ResearchDomain | None = None,
) -> tuple[Any, str]:
    """
    Create an orchestrator instance.

    Unified Architecture (SPEC-16): All users get Advanced Mode.
    Backend auto-selects: OpenAI (if key) → HuggingFace (free fallback).

    Args:
        use_mock: If True, use MockJudgeHandler (no API key needed)
        mode: Orchestrator mode (default "advanced", "hierarchical" for sub-iteration)
        user_api_key: Optional user-provided API key (BYOK) - auto-detects provider
        domain: Research domain (defaults to "sexual_health")

    Returns:
        Tuple of (Orchestrator instance, backend_name)
    """
    # Create orchestrator config
    config = OrchestratorConfig(
        max_iterations=10,
        max_results_per_tool=10,
    )

    backend_info = "Unknown"

    # 1. Forced Mock (Unit Testing)
    if use_mock:
        backend_info = "Mock (Testing)"

    # 2. Paid API Key (User provided or Env)
    elif user_api_key and user_api_key.strip():
        if user_api_key.startswith("sk-"):
            backend_info = "Paid API (OpenAI)"
        else:
            raise ConfigurationError("Invalid API key format. Expected sk-... (OpenAI)")

    # 3. Environment API Keys (fallback)
    elif settings.has_openai_key:
        backend_info = "Paid API (OpenAI from env)"

    # 4. Free Tier (HuggingFace Inference)
    else:
        backend_info = "Free Tier (Llama 3.1 / Mistral)"

    orchestrator = create_orchestrator(
        config=config,
        mode=mode,
        api_key=user_api_key,
        domain=domain,
    )

    return orchestrator, backend_info


def _validate_inputs(
    api_key: str | None,
    api_key_state: str | None,
) -> tuple[str | None, bool]:
    """Validate inputs and determine key status.

    Unified Architecture (SPEC-16): Mode is always "advanced".
    Backend auto-selects based on available API keys.

    Returns:
        Tuple of (effective_user_key, has_paid_key)
    """
    # Determine effective key
    user_api_key = (api_key or api_key_state or "").strip() or None

    # Check available keys
    has_openai = settings.has_openai_key
    has_paid_key = has_openai or bool(user_api_key)

    return user_api_key, has_paid_key


async def research_agent(
    message: str,
    history: list[dict[str, Any]],
    domain: str = "sexual_health",
    api_key: str = "",
    api_key_state: str = "",
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> AsyncGenerator[str, None]:
    """
    Gradio chat function that runs the research agent.

    Unified Architecture (SPEC-16): Always uses Advanced Mode.
    Backend auto-selects: OpenAI (if key) → HuggingFace (free fallback).

    Args:
        message: User's research question
        history: Chat history (Gradio format)
        domain: Research domain
        api_key: Optional user-provided API key (BYOK - auto-detects provider)
        api_key_state: Persistent API key state (survives example clicks)
        progress: Gradio progress tracker

    Yields:
        Markdown-formatted responses for streaming
    """
    if not message.strip():
        yield "Please enter a research question."
        return

    # BUG FIX: Handle None values from Gradio example caching
    domain_str = domain or "sexual_health"

    # Validate inputs (SPEC-16: mode is always "advanced")
    user_api_key, has_paid_key = _validate_inputs(api_key, api_key_state)

    if not has_paid_key:
        yield (
            "🤗 **Free Tier**: Using HuggingFace Inference (Llama 3.1 / Mistral) for AI analysis.\n"
            "For premium models, enter an OpenAI API key below.\n\n"
        )

    # Run the agent and stream events
    response_parts: list[str] = []
    streaming_buffer = ""  # Buffer for accumulating streaming tokens

    try:
        # use_mock=False - let configure_orchestrator decide based on available keys
        # SPEC-16: mode is always "advanced" (unified architecture)
        orchestrator, backend_name = configure_orchestrator(
            use_mock=False,
            mode="advanced",
            user_api_key=user_api_key,
            domain=domain_str,
        )

        # Immediate backend info + loading feedback so user knows something is happening
        # Use replace to get "Sexual Health" instead of "Sexual_Health" from .title()
        domain_display = domain_str.replace("_", " ").title()
        yield (
            f"🧠 **Backend**: {backend_name} | **Domain**: {domain_display}\n\n"
            "⏳ **Processing...** Searching PubMed, ClinicalTrials.gov, Europe PMC, OpenAlex...\n"
        )

        async for event in orchestrator.run(message):
            # Update progress bar
            if event.type == "started":
                progress(0, desc="Starting research...")
            elif event.type == "thinking":
                progress(0.1, desc="Multi-agent reasoning...")
            elif event.type == "progress":
                # Calculate progress percentage (fallback to 0.15 for events without iteration)
                p = 0.15
                max_iters = getattr(orchestrator, "_max_rounds", None) or getattr(
                    getattr(orchestrator, "config", None), "max_iterations", 10
                )
                if event.iteration:
                    # Map 0..max to 0.2..0.9
                    p = 0.2 + (0.7 * (min(event.iteration, max_iters) / max_iters))
                progress(p, desc=event.message)

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
                response_parts.append(event.message)
                yield "\n\n".join(response_parts)
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
    additional_inputs_accordion = gr.Accordion(label="⚙️ API Key (Free tier works!)", open=False)

    # BUG FIX: Add gr.State for API key persistence across example clicks
    api_key_state = gr.State("")

    # 1. Unwrapped ChatInterface (Fixes Accordion Bug)
    # NOTE: Using inline styles on each element because HR breaks text-align inheritance
    description = (
        "<div style='text-align: center;'>"
        "<em>AI-Powered Research Agent — searches PubMed, "
        "ClinicalTrials.gov, Europe PMC & OpenAlex</em><br><br>"
        "Deep research for sexual wellness, ED treatments, hormone therapy, "
        "libido, and reproductive health - for all genders."
        "</div>"
        "<hr style='margin: 1em auto; width: 80%; border: none; "
        "border-top: 1px solid #374151;'>"
        "<div style='text-align: center;'>"
        "<em>Research tool only — not for medical advice.</em><br>"
        "<strong>MCP Server Active</strong>: Connect Claude Desktop to "
        "<code>/gradio_api/mcp/</code>"
        "</div>"
    )

    demo = gr.ChatInterface(
        fn=research_agent,
        title="🍆 DeepBoner",
        description=description,
        examples=[
            # SPEC-16: Mode is always "advanced" (unified architecture)
            # Examples now only need: [question, domain, api_key, api_key_state]
            # NOTE: None values preserve user's current input (not overwritten)
            [
                "What drugs improve female libido post-menopause?",
                "sexual_health",
                None,
                None,
            ],
            [
                "Testosterone therapy for hypoactive sexual desire disorder?",
                "sexual_health",
                None,
                None,
            ],
            [
                "Clinical trials for PDE5 inhibitors alternatives?",
                "sexual_health",
                None,
                None,
            ],
        ],
        # FIX: Prevent examples from auto-submitting (P1_GRADIO_EXAMPLE_CLICK_AUTO_SUBMIT)
        # cache_examples must be False for run_examples_on_click to take effect on HF Spaces
        cache_examples=False,
        run_examples_on_click=False,
        additional_inputs_accordion=additional_inputs_accordion,
        additional_inputs=[
            # SPEC-16: Mode toggle removed - everyone gets Advanced Mode
            # Backend auto-selects: OpenAI (if key) → HuggingFace (free fallback)
            gr.Dropdown(
                choices=[d.value for d in ResearchDomain],
                value="sexual_health",
                label="Research Domain",
                info="DeepBoner specializes in sexual health research",
                visible=False,  # Hidden - only sexual_health supported
            ),
            gr.Textbox(
                label="🔑 API Key (Optional)",
                placeholder="sk-... (OpenAI)",
                type="password",
                info="Leave empty for free tier. Auto-detects provider from key prefix.",
                elem_classes=["api-key-input"],
            ),
            api_key_state,  # Hidden state component for persistence
        ],
    )

    return demo, additional_inputs_accordion


def main() -> None:
    """Run the Gradio app with MCP server enabled."""
    warmup_services()  # Phase 2: Pre-warm services
    demo, _ = create_demo()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),  # nosec B104
        server_port=7860,
        share=False,
        mcp_server=True,
        ssr_mode=False,  # Fix for intermittent loading/hydration issues in HF Spaces
        css=CUSTOM_CSS,  # Moved here for Gradio 6.0 support
    )


if __name__ == "__main__":
    main()
