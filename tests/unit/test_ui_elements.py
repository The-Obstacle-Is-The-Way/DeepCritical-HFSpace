import gradio as gr

from src.app import create_demo


def test_examples_include_advanced_mode():
    """Verify that one example entry uses 'advanced' mode."""
    demo, _ = create_demo()
    assert any(
        example[1] == "advanced" for example in demo.examples
    ), "Expected at least one example to be 'advanced' mode"


def test_accordion_label_updated():
    """Verify the accordion label reflects the new, concise text."""
    _, accordion = create_demo()
    assert (
        accordion.label == "⚙️ Mode & API Key (Free tier works!)"
    ), "Accordion label not updated to '⚙️ Mode & API Key (Free tier works!)'"


def test_orchestrator_mode_info_text_updated():
    """Verify the Orchestrator Mode info text contains the new emojis and phrasing."""
    demo, _ = create_demo()
    # Assuming additional_inputs is a list and the Radio is the first element
    orchestrator_radio = demo.additional_inputs[0]
    expected_info = "⚡ Simple: Free/Any | 🔬 Advanced: OpenAI (Deep Research)"
    assert isinstance(
        orchestrator_radio, gr.Radio
    ), "Expected first additional input to be gr.Radio"
    assert (
        orchestrator_radio.info == expected_info
    ), "Orchestrator Mode info text not updated correctly"
