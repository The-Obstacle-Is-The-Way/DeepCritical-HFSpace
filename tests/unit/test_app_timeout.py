"""Tests for app timeout and history preservation."""

from unittest.mock import MagicMock, patch

import pytest

from src.app import research_agent
from src.utils.models import AgentEvent


async def async_gen(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_complete_event_preserves_history():
    """
    Verify that a 'complete' event (like timeout) appends to the history
    instead of replacing it.
    """
    # Mock events: Progress -> Progress -> Complete
    mock_events = [
        AgentEvent(type="thinking", message="Step 1: Thinking...", iteration=0),
        AgentEvent(type="search_complete", message="Step 2: Found data", iteration=1),
        AgentEvent(type="complete", message="Timeout: Synthesizing...", iteration=1),
    ]

    # Create a mock orchestrator that yields these events
    mock_orchestrator = MagicMock()
    # The run method should return an async generator
    mock_orchestrator.run.side_effect = lambda msg: async_gen(mock_events)

    # Patch configure_orchestrator to return our mock
    with patch("src.app.configure_orchestrator") as mock_config:
        mock_config.return_value = (mock_orchestrator, "Mock Backend")

        # Run the agent
        results = []
        async for output in research_agent("test query", [], "simple"):
            results.append(output)

        # The final output should contain the accumulated history AND the timeout message
        final_output = results[-1]

        # Check for preservation
        assert "Step 1: Thinking..." in final_output
        assert "Step 2: Found data" in final_output
        assert "Timeout: Synthesizing..." in final_output
