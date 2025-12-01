"""Tests for Magentic Orchestrator termination guarantee."""

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if agent_framework not installed (optional dep)
# MUST come before any agent_framework imports
pytest.importorskip("agent_framework")

from agent_framework import MagenticAgentMessageEvent  # noqa: E402

from src.orchestrators.advanced import AdvancedOrchestrator as MagenticOrchestrator  # noqa: E402
from src.utils.models import AgentEvent  # noqa: E402


class MockChatMessage:
    def __init__(self, content):
        self.content = content
        self.role = "assistant"

    @property
    def text(self):
        return self.content


@pytest.fixture
def mock_magentic_requirements():
    """Mock requirements check."""
    with patch("src.orchestrators.advanced.check_magentic_requirements"):
        yield


@pytest.mark.asyncio
async def test_termination_event_emitted_on_stream_end(mock_magentic_requirements):
    """
    Verify that a termination event is emitted when the workflow stream ends
    without a MagenticFinalResultEvent (e.g. max rounds reached).
    """
    orchestrator = MagenticOrchestrator(max_rounds=2)

    # Use real event class
    mock_message = MockChatMessage("Thinking...")
    mock_agent_event = MagenticAgentMessageEvent(agent_id="SearchAgent", message=mock_message)

    # Mock the workflow and its run_stream method
    mock_workflow = MagicMock()

    # Create an async generator for run_stream
    async def mock_stream(task):
        # Yield the real message event
        yield mock_agent_event
        # STOP HERE - No FinalResultEvent

    mock_workflow.run_stream = mock_stream

    # Mock _build_workflow to return our mock workflow
    with patch.object(orchestrator, "_build_workflow", return_value=mock_workflow):
        events = []
        async for event in orchestrator.run("Research query"):
            events.append(event)

        for i, e in enumerate(events):
            print(f"Event {i}: {e.type} - {e.message}")

        assert len(events) >= 2
        assert events[0].type == "started"

        # Verify the message event was processed
        # Depending on _process_event logic, MagenticAgentMessageEvent might map to different types
        # We assume it maps to something valid or we just check presence.
        assert any("Thinking..." in e.message for e in events)

        # THE CRITICAL CHECK: Did we get the fallback termination event?
        last_event = events[-1]
        assert last_event.type == "complete"
        assert "Max iterations reached" in last_event.message
        assert last_event.data.get("reason") == "max_rounds_reached"


@pytest.mark.asyncio
async def test_no_double_termination_event(mock_magentic_requirements):
    """
    Verify that we DO NOT emit a fallback event if the workflow finished normally.
    """
    orchestrator = MagenticOrchestrator()

    mock_workflow = MagicMock()

    with patch.object(orchestrator, "_build_workflow", return_value=mock_workflow):
        # Mock _process_event to simulate a natural completion event
        with patch.object(orchestrator, "_process_event") as mock_process:
            mock_process.side_effect = [
                AgentEvent(type="thinking", message="Working...", iteration=1),
                AgentEvent(type="complete", message="Done!", iteration=2),
            ]

            async def mock_stream_with_yields(task):
                yield "raw_event_1"
                yield "raw_event_2"

            mock_workflow.run_stream = mock_stream_with_yields

            events = []
            async for event in orchestrator.run("Research query"):
                events.append(event)

            assert events[-1].message == "Done!"
            assert events[-1].type == "complete"

            # Verify we didn't get a SECOND "Max iterations reached" event
            fallback_events = [e for e in events if "Max iterations reached" in e.message]
            assert len(fallback_events) == 0


@pytest.mark.asyncio
async def test_termination_on_timeout(mock_magentic_requirements):
    """
    Verify that a termination event is emitted when the workflow times out.
    """
    orchestrator = MagenticOrchestrator()

    mock_workflow = MagicMock()

    # Simulate a stream that times out (raises TimeoutError)
    async def mock_stream_raises(task):
        # Yield one event before timing out
        yield MagenticAgentMessageEvent(
            agent_id="SearchAgent", message=MockChatMessage("Working...")
        )
        raise TimeoutError()

    mock_workflow.run_stream = mock_stream_raises

    with patch.object(orchestrator, "_build_workflow", return_value=mock_workflow):
        events = []
        async for event in orchestrator.run("Research query"):
            events.append(event)

        # Check for progress/normal events
        assert any("Working..." in e.message for e in events)

        # Check for timeout completion
        completion_events = [e for e in events if e.type == "complete"]
        assert len(completion_events) > 0
        last_event = completion_events[-1]

        # New behavior: synthesis is attempted on timeout
        # The message contains the report, so we check the reason code
        # In unit tests without API keys, synthesis will fail -> "timeout_synthesis_failed"
        assert last_event.data.get("reason") in (
            "timeout",
            "timeout_synthesis",
            "timeout_synthesis_failed",  # Expected in unit tests (no API key)
        )
