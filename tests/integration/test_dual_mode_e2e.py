"""End-to-End Integration Tests for Dual-Mode Architecture."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from src.orchestrator_factory import create_orchestrator
from src.utils.models import Citation, Evidence, OrchestratorConfig


@pytest.fixture
def mock_search_handler():
    handler = MagicMock()
    handler.execute = AsyncMock(
        return_value=[
            Evidence(
                citation=Citation(
                    title="Test Paper", url="http://test", date="2024", source="pubmed"
                ),
                content="Metformin increases lifespan in mice.",
            )
        ]
    )
    return handler


@pytest.fixture
def mock_judge_handler():
    handler = MagicMock()
    # Mock return value of assess
    assessment = MagicMock()
    assessment.sufficient = True
    assessment.recommendation = "synthesize"
    handler.assess = AsyncMock(return_value=assessment)
    return handler


@pytest.mark.asyncio
async def test_simple_mode_e2e(mock_search_handler, mock_judge_handler):
    """Test Simple Mode Orchestration flow."""
    orch = create_orchestrator(
        search_handler=mock_search_handler,
        judge_handler=mock_judge_handler,
        mode="simple",
        config=OrchestratorConfig(max_iterations=1),
    )

    # Run
    results = []
    async for event in orch.run("Test query"):
        results.append(event)

    assert len(results) > 0
    assert mock_search_handler.execute.called
    assert mock_judge_handler.assess.called


@pytest.mark.asyncio
async def test_advanced_mode_explicit_instantiation():
    """Test explicit Advanced Mode instantiation (not auto-detect).

    This tests the explicit mode="advanced" path, verifying that
    MagenticOrchestrator can be instantiated when explicitly requested.
    The settings patch ensures any internal checks pass.
    """
    with patch("src.orchestrator_factory.settings") as mock_settings:
        # Settings patch ensures factory checks pass (even though mode is explicit)
        mock_settings.has_openai_key = True

        with patch("src.agents.magentic_agents.OpenAIChatClient"):
            # Mock agent creation to avoid real API calls during init
            with (
                patch("src.orchestrator_magentic.create_search_agent"),
                patch("src.orchestrator_magentic.create_judge_agent"),
                patch("src.orchestrator_magentic.create_hypothesis_agent"),
                patch("src.orchestrator_magentic.create_report_agent"),
            ):
                # Explicit mode="advanced" - tests the explicit path, not auto-detect
                orch = create_orchestrator(mode="advanced")
                assert orch is not None
