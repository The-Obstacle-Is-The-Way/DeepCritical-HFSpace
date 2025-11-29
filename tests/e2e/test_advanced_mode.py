from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if agent_framework is not installed
agent_framework = pytest.importorskip("agent_framework")
from agent_framework import MagenticAgentMessageEvent, MagenticFinalResultEvent

from src.orchestrator_magentic import MagenticOrchestrator


class MockChatMessage:
    def __init__(self, content):
        self.content = content

    @property
    def text(self):
        return self.content


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_advanced_mode_completes_mocked():
    """Verify Advanced mode runs without crashing (mocked workflow)."""

    # Initialize orchestrator (mocking requirements check)
    with patch("src.orchestrator_magentic.check_magentic_requirements"):
        orchestrator = MagenticOrchestrator(max_rounds=5)

    # Mock the workflow
    mock_workflow = MagicMock()

    # Create fake events
    # 1. Search Agent runs
    mock_msg_1 = MockChatMessage("Found 5 papers on PubMed")
    event1 = MagenticAgentMessageEvent(agent_id="SearchAgent", message=mock_msg_1)

    # 2. Report Agent finishes
    mock_result_msg = MockChatMessage("# Final Report\n\nFindings...")
    event2 = MagenticFinalResultEvent(message=mock_result_msg)

    async def mock_stream(task):
        yield event1
        yield event2

    mock_workflow.run_stream = mock_stream

    # Patch dependencies:
    # _build_workflow: Returns our mock
    # init_magentic_state: Avoids DB calls
    # _init_embedding_service: Avoids loading embeddings
    with (
        patch.object(orchestrator, "_build_workflow", return_value=mock_workflow),
        patch("src.orchestrator_magentic.init_magentic_state"),
        patch.object(orchestrator, "_init_embedding_service", return_value=None),
    ):
        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Check events
        types = [e.type for e in events]
        assert "started" in types
        assert "thinking" in types
        assert "search_complete" in types  # Mapped from SearchAgent
        assert "progress" in types  # Added in SPEC_01
        assert "complete" in types

        complete_event = next(e for e in events if e.type == "complete")
        assert "Final Report" in complete_event.message
