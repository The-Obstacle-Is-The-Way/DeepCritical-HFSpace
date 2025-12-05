from unittest.mock import MagicMock, patch

import pytest

from src.orchestrators.advanced import AdvancedOrchestrator


@pytest.mark.asyncio
@pytest.mark.unit
async def test_advanced_initialization_events():
    """Verify granular progress events are emitted during initialization."""
    # Mock dependencies
    with (
        patch("src.orchestrators.advanced.AdvancedOrchestrator._init_embedding_service"),
        patch("src.orchestrators.advanced.init_magentic_state"),
        patch("src.orchestrators.advanced.AdvancedOrchestrator._build_workflow") as mock_build,
    ):  # Bypass check
        # Setup mocks
        mock_workflow = MagicMock()

        # Mock run_stream to return an empty async iterator
        async def mock_stream(task):
            # Just yield nothing effectively, we break before this anyway
            if False:
                yield None

        mock_workflow.run_stream = mock_stream
        mock_build.return_value = mock_workflow

        # Initialize orchestrator with dummy key to bypass requirement check in __init__
        orch = AdvancedOrchestrator(api_key="sk-dummy")

        # Run
        events = []
        try:
            async for event in orch.run("test query"):
                events.append(event)
                # We want to capture up to the 'thinking' event which comes after init
                if event.type == "thinking":
                    break
        except Exception as e:
            pytest.fail(f"Orchestrator run failed: {e}")

        # Verify sequence
        messages = [e.message for e in events]
        types = [e.type for e in events]

        # Expected sequence:
        # 1. started
        # 2. progress (Loading embedding...)
        # 3. progress (Initializing research...)
        # 4. progress (Building agent team...)
        # 5. thinking

        assert len(messages) >= 5, "Not enough events emitted"

        assert messages[0].startswith("Starting research")
        assert messages[1] == "Loading embedding service (LlamaIndex/ChromaDB)..."
        assert messages[2] == "Initializing research memory..."
        assert messages[3] == "Building agent team (Search, Judge, Hypothesis, Report)..."
        assert messages[4].startswith("Multi-agent reasoning")

        assert types[1] == "progress"
        assert types[2] == "progress"
        assert types[3] == "progress"
