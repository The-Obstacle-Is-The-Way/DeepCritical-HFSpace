"""Test for AdvancedOrchestrator event processing (P1 Bug)."""

from unittest.mock import MagicMock

import pytest
from agent_framework import MAGENTIC_EVENT_TYPE_ORCHESTRATOR

from src.orchestrators.advanced import AdvancedOrchestrator


class MockOrchestratorEvent:
    """Mock event that mimics the new orchestrator event structure."""

    def __init__(self, kind: str, message: str):
        self.type = MAGENTIC_EVENT_TYPE_ORCHESTRATOR
        self.kind = kind
        self.message = MagicMock()
        self.message.text = message


@pytest.mark.unit
class TestAdvancedEventProcessing:
    """Test event processing logic in AdvancedOrchestrator."""

    @pytest.fixture
    def orchestrator(self) -> AdvancedOrchestrator:
        """Create an orchestrator instance with mocks."""
        # Bypass __init__ logic that requires keys/env vars
        orch = AdvancedOrchestrator.__new__(AdvancedOrchestrator)
        # Minimal setup
        orch._max_rounds = 5
        orch._timeout_seconds = 300.0
        return orch

    def test_filters_internal_task_ledger_events(self, orchestrator: AdvancedOrchestrator) -> None:
        """
        Bug P1: Internal 'task_ledger' events should be filtered out.

        Current behavior: Returns AgentEvent(type='judging', message='Manager (task_ledger): ...')
        Desired behavior: Returns None (filtered)
        """
        # Create a raw internal framework event
        raw_event = MockOrchestratorEvent(
            kind="task_ledger",
            message="We are working to address the following user request: Research sildenafil...",
        )

        # Process the event
        result = orchestrator._process_event(raw_event, iteration=1)

        # FAIL if the event is NOT filtered (i.e., if it returns an event)
        assert result is None, f"Should filter 'task_ledger' events, but got: {result}"

    def test_filters_internal_instruction_events(self, orchestrator: AdvancedOrchestrator) -> None:
        """
        Bug P1: Internal 'instruction' events should be filtered out.

        Current behavior: Returns AgentEvent(type='judging', message='Manager (instruction): ...')
        Desired behavior: Returns None (filtered)
        """
        raw_event = MockOrchestratorEvent(
            kind="instruction", message="Conduct targeted searches on PubMed..."
        )

        result = orchestrator._process_event(raw_event, iteration=1)

        assert result is None, f"Should filter 'instruction' events, but got: {result}"

    def test_transforms_user_task_events(self, orchestrator: AdvancedOrchestrator) -> None:
        """
        Bug P1: 'user_task' events should be transformed to user-friendly messages.

        Current behavior: 'Manager (user_task): Research...' (truncated, type='judging')
        Desired behavior: 'Manager assigning research task...' (type='progress')
        """
        raw_event = MockOrchestratorEvent(
            kind="user_task",
            message="Research sexual health and wellness interventions for: sildenafil mechanism",
        )

        result = orchestrator._process_event(raw_event, iteration=1)

        assert result is not None
        assert result.type == "progress"  # NOT "judging"
        assert "Manager assigning research task" in result.message
        # Should use the generic friendly message
        assert "sildenafil mechanism" not in result.message

    def test_prevents_mid_sentence_truncation(self, orchestrator: AdvancedOrchestrator) -> None:
        """
        Bug P1: Long messages should be smart-truncated at sentence boundaries.

        Tests _smart_truncate directly to ensure regression protection.
        The function truncates at sentence boundary if period is after halfway point.
        """
        # First sentence ends at position ~55, which is > 50 (100//2)
        long_text = (
            "This is a longer first sentence that ends past the midpoint. "
            "Second sentence continues with more text that would be cut."
        )

        # Call the helper directly to test its behavior explicitly
        truncated = orchestrator._smart_truncate(long_text, max_len=100)

        # Should truncate at the end of the first sentence (period > max_len//2)
        assert truncated.endswith("midpoint.")
        assert "Second sentence" not in truncated
        assert len(truncated) <= 100

    def test_smart_truncate_word_boundary_fallback(
        self, orchestrator: AdvancedOrchestrator
    ) -> None:
        """Test that truncation falls back to word boundary when no sentence end."""
        # No sentence ending in the first 80 chars
        long_text = "This is a very long text without any sentence ending in the limit"

        truncated = orchestrator._smart_truncate(long_text, max_len=50)

        # Should end with "..." and not cut mid-word
        assert truncated.endswith("...")
        assert len(truncated) <= 53  # 50 + "..."
