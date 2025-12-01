"""Test that Gradio example caching doesn't crash with None parameters."""

from unittest.mock import MagicMock

import pytest

from src.utils.models import AgentEvent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_research_agent_handles_none_parameters():
    """
    Test that research_agent handles None parameters gracefully.

    This simulates Gradio's example caching behavior where missing
    example columns are passed as None instead of using default values.

    Bug: https://huggingface.co/spaces/MCP-1st-Birthday/DeepBoner crashed
    because api_key=None and api_key_state=None caused .strip() to fail.
    """
    # Mock the orchestrator to avoid real API calls
    import src.app as app_module
    from src.app import research_agent

    mock_orchestrator = MagicMock()

    async def mock_run(query):
        yield AgentEvent(type="complete", message="Test complete", iteration=1)

    mock_orchestrator.run = mock_run

    original_configure = app_module.configure_orchestrator
    app_module.configure_orchestrator = MagicMock(return_value=(mock_orchestrator, "Mock"))

    try:
        # This should NOT raise AttributeError: 'NoneType' object has no attribute 'strip'
        results = []
        # SPEC-16: mode parameter removed (unified architecture)
        async for result in research_agent(
            message="test query",
            history=[],
            api_key=None,  # Simulating Gradio passing None
            api_key_state=None,  # Simulating Gradio passing None
        ):
            results.append(result)

        # If we get here without AttributeError, the fix works
        assert len(results) > 0, "Should have yielded at least one result"

    finally:
        app_module.configure_orchestrator = original_configure


@pytest.mark.unit
@pytest.mark.asyncio
async def test_research_agent_handles_empty_string_parameters():
    """Test that empty strings (the expected default) also work."""
    import src.app as app_module
    from src.app import research_agent

    mock_orchestrator = MagicMock()

    async def mock_run(query):
        yield AgentEvent(type="complete", message="Test complete", iteration=1)

    mock_orchestrator.run = mock_run

    original_configure = app_module.configure_orchestrator
    app_module.configure_orchestrator = MagicMock(return_value=(mock_orchestrator, "Mock"))

    try:
        results = []
        # SPEC-16: mode parameter removed (unified architecture)
        async for result in research_agent(
            message="test query",
            history=[],
            api_key="",  # Normal empty string
            api_key_state="",  # Normal empty string
        ):
            results.append(result)

        assert len(results) > 0

    finally:
        app_module.configure_orchestrator = original_configure
