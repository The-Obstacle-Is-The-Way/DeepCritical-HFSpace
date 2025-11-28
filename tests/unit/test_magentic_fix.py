"""Tests for Magentic Orchestrator fixes."""

from unittest.mock import MagicMock, patch

import pytest
from agent_framework import MagenticFinalResultEvent

from src.orchestrator_magentic import MagenticOrchestrator


class MockChatMessage:
    """Simulates the buggy ChatMessage that returns itself as text or has complex content."""

    def __init__(self, content_str: str) -> None:
        self.content_str = content_str
        self.role = "assistant"

    @property
    def text(self) -> "MockChatMessage":
        # Simulate the bug: .text returns the object itself or a repr string
        return self

    @property
    def content(self) -> str:
        # The fix plan says we should look for .content
        return self.content_str

    def __repr__(self) -> str:
        return "<ChatMessage object at 0xMOCK>"

    def __str__(self) -> str:
        return "<ChatMessage object at 0xMOCK>"


@pytest.fixture
def mock_magentic_requirements():
    """Mock the API key check so tests run in CI without OPENAI_API_KEY."""
    with patch("src.orchestrator_magentic.check_magentic_requirements"):
        yield


class TestMagenticFixes:
    """Tests for the Magentic mode fixes."""

    def test_process_event_extracts_text_correctly(self, mock_magentic_requirements) -> None:
        """
        Test that _process_event correctly extracts text from a ChatMessage.

        Verifies fix for bug where .text returns the object itself.
        """
        orchestrator = MagenticOrchestrator()

        # Create a mock message that mimics the bug
        buggy_message = MockChatMessage("Final Report Content")
        event = MagenticFinalResultEvent(message=buggy_message)  # type: ignore[arg-type]

        # Process the event
        # We expect the fix to get "Final Report Content" instead of object repr
        result_event = orchestrator._process_event(event, iteration=1)

        assert result_event is not None
        assert result_event.type == "complete"
        assert result_event.message == "Final Report Content"

    def test_max_rounds_configuration(self, mock_magentic_requirements) -> None:
        """Test that max_rounds is correctly passed to the orchestrator."""
        orchestrator = MagenticOrchestrator(max_rounds=25)
        assert orchestrator._max_rounds == 25

        # Also verify it's used in _build_workflow
        # Mock all the agent creation and OpenAI client calls
        with (
            patch("src.orchestrator_magentic.create_search_agent") as mock_search,
            patch("src.orchestrator_magentic.create_judge_agent") as mock_judge,
            patch("src.orchestrator_magentic.create_hypothesis_agent") as mock_hypo,
            patch("src.orchestrator_magentic.create_report_agent") as mock_report,
            patch("src.orchestrator_magentic.OpenAIChatClient") as mock_client,
            patch("src.orchestrator_magentic.MagenticBuilder") as mock_builder,
        ):
            # Setup mocks
            mock_search.return_value = MagicMock()
            mock_judge.return_value = MagicMock()
            mock_hypo.return_value = MagicMock()
            mock_report.return_value = MagicMock()
            mock_client.return_value = MagicMock()

            # Mock the builder chain
            mock_chain = mock_builder.return_value.participants.return_value
            mock_chain.with_standard_manager.return_value.build.return_value = MagicMock()

            orchestrator._build_workflow()

            # Check that max_round_count was passed as 25
            participants_mock = mock_builder.return_value.participants.return_value
            participants_mock.with_standard_manager.assert_called_once()
            call_kwargs = participants_mock.with_standard_manager.call_args.kwargs
            assert call_kwargs["max_round_count"] == 25
