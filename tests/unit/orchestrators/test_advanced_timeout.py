from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrators.advanced import AdvancedOrchestrator
from src.orchestrators.factory import create_orchestrator
from src.utils.config import settings


@pytest.mark.asyncio
async def test_timeout_synthesizes_evidence():
    """Timeout should produce synthesis, not empty message."""
    mock_client = MagicMock()
    orchestrator = AdvancedOrchestrator(
        max_rounds=1,
        timeout_seconds=0.01,
        chat_client=mock_client,
    )

    async def slow_stream(*args, **kwargs):
        import asyncio

        await asyncio.sleep(0.1)
        yield MagicMock()

    mock_workflow = MagicMock()
    mock_workflow.run_stream = slow_stream

    # Mock dependencies used inside the timeout block
    with (
        patch.object(orchestrator, "_build_workflow", return_value=mock_workflow),
        patch("src.orchestrators.advanced.init_magentic_state"),
        patch("src.agents.state.get_magentic_state") as mock_get_state,
        patch("src.agents.magentic_agents.create_report_agent") as mock_create_agent,
    ):
        # Setup mock state and memory
        mock_memory = AsyncMock()
        mock_memory.get_context_summary.return_value = "Mocked Evidence Summary"
        mock_state = MagicMock()
        mock_state.memory = mock_memory
        mock_get_state.return_value = mock_state

        # Setup mock ReportAgent
        # ChatAgent.run() returns AgentRunResponse with .text property
        mock_report_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Final Synthesized Report"
        mock_report_agent.run.return_value = mock_response
        mock_create_agent.return_value = mock_report_agent

        events = []
        async for e in orchestrator.run("test query"):
            events.append(e)

        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) > 0
        complete_event = complete_events[-1]

        # Verify synthesis happened
        assert complete_event.message == "Final Synthesized Report"
        assert complete_event.data["reason"] == "timeout_synthesis"

        # Verify mocks were called
        mock_memory.get_context_summary.assert_called_once()
        mock_create_agent.assert_called_once()
        mock_report_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_uses_advanced_max_rounds():
    """Factory should use settings.advanced_max_rounds for advanced mode."""
    assert settings.advanced_max_rounds == 5

    # Mock the internal helper that returns the class
    with patch("src.orchestrators.factory._get_advanced_orchestrator_class") as mock_get_cls:
        # Create a mock class that acts like AdvancedOrchestrator
        mock_cls = MagicMock()
        mock_get_cls.return_value = mock_cls

        create_orchestrator(
            mode="advanced",
            api_key="sk-test",
        )

        # Verify the mock class was instantiated with correct max_rounds
        _, kwargs = mock_cls.call_args
        assert kwargs["max_rounds"] == 5
