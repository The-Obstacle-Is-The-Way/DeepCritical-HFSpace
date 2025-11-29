"""Test that streaming event handling is fixed (no token-by-token spam)."""

from unittest.mock import MagicMock

import pytest

from src.utils.models import AgentEvent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_events_are_buffered_not_spammed():
    """
    Verify that streaming events are buffered, not yielded individually.

    This test validates the fix for Bug 1: Token-by-Token Streaming Spam.
    Before the fix, each token would create a separate yield, resulting in O(N²) spam.
    After the fix, streaming tokens are buffered and only yielded once.
    """
    # Import here to avoid circular dependencies
    from src.app import research_agent

    # Mock orchestrator
    mock_orchestrator = MagicMock()

    # Simulate streaming events (like LLM token-by-token output)
    streaming_events = [
        AgentEvent(type="started", message="Starting research", iteration=0),
        AgentEvent(type="streaming", message="This", iteration=1),
        AgentEvent(type="streaming", message=" is", iteration=1),
        AgentEvent(type="streaming", message=" a", iteration=1),
        AgentEvent(type="streaming", message=" test", iteration=1),
        AgentEvent(type="complete", message="Final answer: This is a test", iteration=1),
    ]

    # Create async generator that yields events
    async def mock_run(query):
        for event in streaming_events:
            yield event

    mock_orchestrator.run = mock_run

    # Mock configure_orchestrator to return our mock
    import src.app as app_module

    original_configure = app_module.configure_orchestrator
    app_module.configure_orchestrator = MagicMock(return_value=(mock_orchestrator, "Test Backend"))

    try:
        # Run the research agent
        results = []
        async for result in research_agent("test query", [], mode="simple", api_key=""):
            results.append(result)

        # Verify that we DO see streaming updates (for UX responsiveness)
        # But we don't want O(N^2) growth of the persisted list.

        # We expect results to contain the streaming updates
        assert len(results) > 0, "Should have yielded results"

        # Check that we see the accumulated message
        assert any(
            "📡 **STREAMING**: This is a test" in r for r in results
        ), "Buffer didn't accumulate correctly"

        # The critical check for the "Spam" bug:
        # In the spam bug, the output grew like:
        # "Stream: T"
        # "Stream: T\nStream: h"
        # "Stream: T\nStream: h\nStream: i"
        #
        # In the fixed version, it should look like:
        # "Stream: T"
        # "Stream: Th"
        # "Stream: Thi"
        # (Replacing the last line, not adding new lines)

        for res in results:
            # Count occurrences of "📡 **STREAMING**:": in a single result string
            # It should appear AT MOST once
            # (unless we have multiple distinct streaming blocks)
            streaming_markers = res.count("📡 **STREAMING**:")
            assert streaming_markers <= 1, (
                f"Found multiple streaming markers in single response: {res}\n"
                "This indicates we are appending new lines instead of updating in place."
            )

        # The final result should be the complete message
        assert any("Final answer" in r for r in results), "Missing final complete message"

    finally:
        # Restore original function
        app_module.configure_orchestrator = original_configure


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_key_state_parameter_exists():
    """
    Verify that api_key_state parameter was added to research_agent.

    This validates the fix for Bug 2: API Key Persistence.
    """
    import inspect

    from src.app import research_agent

    # Get function signature
    sig = inspect.signature(research_agent)
    params = list(sig.parameters.keys())

    # Verify api_key_state parameter exists
    assert "api_key_state" in params, "api_key_state parameter missing from research_agent"

    # Verify it's after api_key
    api_key_idx = params.index("api_key")
    api_key_state_idx = params.index("api_key_state")
    assert api_key_state_idx > api_key_idx, "api_key_state should come after api_key"
