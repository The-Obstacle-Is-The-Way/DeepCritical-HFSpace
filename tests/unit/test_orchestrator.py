"""Unit tests for Orchestrator."""

from unittest.mock import AsyncMock, patch

import pytest

from src.orchestrators import Orchestrator
from src.utils.models import (
    AgentEvent,
    AssessmentDetails,
    Citation,
    Evidence,
    JudgeAssessment,
    OrchestratorConfig,
    SearchResult,
)


class TestOrchestrator:
    """Tests for Orchestrator."""

    @pytest.fixture
    def mock_search_handler(self):
        """Create a mock search handler."""
        handler = AsyncMock()
        handler.execute = AsyncMock(
            return_value=SearchResult(
                query="test",
                evidence=[
                    Evidence(
                        content="Test content",
                        citation=Citation(
                            source="pubmed",
                            title="Test Title",
                            url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                            date="2024-01-01",
                        ),
                    ),
                ],
                sources_searched=["pubmed"],
                total_found=1,
                errors=[],
            )
        )
        return handler

    @pytest.fixture
    def mock_judge_sufficient(self):
        """Create a mock judge that returns sufficient."""
        handler = AsyncMock()
        handler.assess = AsyncMock(
            return_value=JudgeAssessment(
                details=AssessmentDetails(
                    mechanism_score=8,
                    mechanism_reasoning="Good mechanism",
                    clinical_evidence_score=7,
                    clinical_reasoning="Good clinical",
                    drug_candidates=["Drug A"],
                    key_findings=["Finding 1"],
                ),
                sufficient=True,
                confidence=0.85,
                recommendation="synthesize",
                next_search_queries=[],
                reasoning="Evidence is sufficient",
            )
        )
        return handler

    @pytest.fixture
    def mock_judge_insufficient(self):
        """Create a mock judge that returns insufficient."""
        handler = AsyncMock()
        handler.assess = AsyncMock(
            return_value=JudgeAssessment(
                details=AssessmentDetails(
                    mechanism_score=4,
                    mechanism_reasoning="Weak mechanism",
                    clinical_evidence_score=3,
                    clinical_reasoning="Weak clinical",
                    drug_candidates=[],
                    key_findings=[],
                ),
                sufficient=False,
                confidence=0.3,
                recommendation="continue",
                next_search_queries=["more specific query"],
                reasoning="Need more evidence to make a decision.",
            )
        )
        return handler

    @pytest.mark.asyncio
    async def test_orchestrator_completes_with_sufficient_evidence(
        self,
        mock_search_handler,
        mock_judge_sufficient,
    ):
        """Orchestrator should complete when evidence is sufficient."""
        config = OrchestratorConfig(max_iterations=5)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_sufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have started, searched, judged, and completed
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "searching" in event_types
        assert "search_complete" in event_types
        assert "judging" in event_types
        assert "judge_complete" in event_types
        assert "complete" in event_types

        # Should only have 1 iteration
        complete_event = next(e for e in events if e.type == "complete")
        assert complete_event.iteration == 1

    @pytest.mark.asyncio
    async def test_orchestrator_loops_when_insufficient(
        self,
        mock_search_handler,
        mock_judge_insufficient,
    ):
        """Orchestrator should loop when evidence is insufficient."""
        config = OrchestratorConfig(max_iterations=3)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have looping events
        event_types = [e.type for e in events]
        assert event_types.count("looping") >= 2  # noqa: PLR2004

        # Should hit max iterations
        complete_event = next(e for e in events if e.type == "complete")
        assert complete_event.data.get("max_reached") is True

    @pytest.mark.asyncio
    async def test_orchestrator_respects_max_iterations(
        self,
        mock_search_handler,
        mock_judge_insufficient,
    ):
        """Orchestrator should stop at max_iterations."""
        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search_handler,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should have exactly 2 iterations
        max_iteration = max(e.iteration for e in events)
        assert max_iteration == 2  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_orchestrator_handles_search_error(self):
        """Orchestrator should handle search errors gracefully."""
        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(side_effect=Exception("Search failed"))

        mock_judge = AsyncMock()
        mock_judge.assess = AsyncMock(
            return_value=JudgeAssessment(
                details=AssessmentDetails(
                    mechanism_score=0,
                    mechanism_reasoning="Not applicable here.",
                    clinical_evidence_score=0,
                    clinical_reasoning="Not applicable here.",
                    drug_candidates=[],
                    key_findings=[],
                ),
                sufficient=False,
                confidence=0.0,
                recommendation="continue",
                next_search_queries=["retry query"],
                reasoning="Search failed, retrying...",
            )
        )

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge,
            config=config,
        )

        events = []
        async for event in orchestrator.run("test query"):
            events.append(event)

        # Should recover and loop despite errors
        event_types = [e.type for e in events]
        assert "error" not in event_types
        assert "looping" in event_types

    @pytest.mark.asyncio
    async def test_orchestrator_deduplicates_evidence(self, mock_judge_insufficient):
        """Orchestrator should deduplicate evidence by URL."""
        # Search returns same evidence each time
        duplicate_evidence = Evidence(
            content="Duplicate content",
            citation=Citation(
                source="pubmed",
                title="Same Title",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",  # Same URL
                date="2024-01-01",
            ),
        )

        mock_search = AsyncMock()
        mock_search.execute = AsyncMock(
            return_value=SearchResult(
                query="test",
                evidence=[duplicate_evidence],
                sources_searched=["pubmed"],
                total_found=1,
                errors=[],
            )
        )

        config = OrchestratorConfig(max_iterations=2)
        orchestrator = Orchestrator(
            search_handler=mock_search,
            judge_handler=mock_judge_insufficient,
            config=config,
        )

        # Force use of local (in-memory) embedding service for test isolation
        # Without this, the test uses persistent LlamaIndex store which has data from previous runs
        with patch("src.utils.service_loader.settings") as mock_settings:
            mock_settings.has_openai_key = False

            events = []
            async for event in orchestrator.run("test query"):
                events.append(event)

        # Second search_complete should show 0 new evidence
        search_complete_events = [e for e in events if e.type == "search_complete"]
        assert len(search_complete_events) == 2  # noqa: PLR2004

        # First iteration should have 1 new
        assert search_complete_events[0].data["new_count"] == 1

        # Second iteration should have 0 new (duplicate)
        assert search_complete_events[1].data["new_count"] == 0


class TestAgentEvent:
    """Tests for AgentEvent."""

    def test_to_markdown(self):
        """AgentEvent should format to markdown correctly."""
        event = AgentEvent(
            type="searching",
            message="Searching for: metformin alzheimer",
            iteration=1,
        )

        md = event.to_markdown()
        assert "🔍" in md
        assert "SEARCHING" in md
        assert "metformin alzheimer" in md

    def test_complete_event_icon(self):
        """Complete event should have celebration icon."""
        event = AgentEvent(
            type="complete",
            message="Done!",
            iteration=3,
        )

        md = event.to_markdown()
        assert "🎉" in md
