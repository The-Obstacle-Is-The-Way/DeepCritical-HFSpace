"""UI element tests for SPEC-16 Unified Architecture."""

import gradio as gr
import pytest

from src.app import create_demo

pytestmark = pytest.mark.unit


def test_no_mode_selector_in_ui():
    """SPEC-16: Mode selector removed - everyone gets Advanced Mode."""
    demo, _ = create_demo()
    # No Radio should exist in additional_inputs
    radios = [inp for inp in demo.additional_inputs if isinstance(inp, gr.Radio)]
    assert len(radios) == 0, "Mode Radio should not exist (SPEC-16: unified architecture)"


def test_accordion_label_updated():
    """Verify the accordion label reflects the new, concise text (no Mode)."""
    _, accordion = create_demo()
    assert accordion.label == "⚙️ API Key (Free tier works!)", (
        f"Accordion label should be '⚙️ API Key (Free tier works!)', got '{accordion.label}'"
    )


def test_examples_have_no_mode():
    """SPEC-16: Examples no longer include mode parameter."""
    demo, _ = create_demo()
    # Examples now have 4 items: [question, domain, api_key, api_key_state]
    for example in demo.examples:
        assert len(example) == 4, (
            f"Examples should have 4 items [question, domain, api_key, api_key_state], "
            f"got {len(example)}: {example}"
        )
        # First item is the question
        assert isinstance(example[0], str) and len(example[0]) > 10, (
            "First example item should be the research question"
        )
        # Second item is domain (not mode!)
        assert example[1] in ("sexual_health", None), (
            f"Second example item should be domain, got: {example[1]}"
        )


def test_api_key_textbox_exists():
    """Verify API key textbox exists in additional inputs."""
    demo, _ = create_demo()
    textboxes = [inp for inp in demo.additional_inputs if isinstance(inp, gr.Textbox)]
    assert len(textboxes) == 1, "Expected exactly one API key textbox"
    assert textboxes[0].label == "🔑 API Key (Optional)", (
        f"API key textbox label should be '🔑 API Key (Optional)', got '{textboxes[0].label}'"
    )
