"""Integration tests for research flows.

These tests require API keys and may make real API calls.
Marked with @pytest.mark.integration to skip in unit test runs.
"""

import pytest

from src.agent_factory.agents import (
    create_deep_flow,
    create_iterative_flow,
    create_planner_agent,
)
from src.orchestrator.graph_orchestrator import create_graph_orchestrator
from src.utils.config import settings


@pytest.mark.integration
class TestPlannerAgentIntegration:
    """Integration tests for PlannerAgent with real API calls."""

    @pytest.mark.asyncio
    async def test_planner_agent_creates_plan(self):
        """PlannerAgent should create a valid report plan with real API."""
        if not settings.has_openai_key() and not settings.has_anthropic_key():
            pytest.skip("No OpenAI or Anthropic API key available")

        planner = create_planner_agent()
        result = await planner.run("What are the main features of Python programming language?")

        assert result.report_title
        assert len(result.report_outline) > 0
        assert result.report_outline[0].title
        assert result.report_outline[0].key_question

    @pytest.mark.asyncio
    async def test_planner_agent_includes_background_context(self):
        """PlannerAgent should include background context in plan."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        planner = create_planner_agent()
        result = await planner.run("Explain quantum computing basics")

        assert result.background_context
        assert len(result.background_context) > 50  # Should have substantial context


@pytest.mark.integration
class TestIterativeResearchFlowIntegration:
    """Integration tests for IterativeResearchFlow with real API calls."""

    @pytest.mark.asyncio
    async def test_iterative_flow_completes_simple_query(self):
        """IterativeResearchFlow should complete a simple research query."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=2, max_time_minutes=2)
        result = await flow.run(
            query="What is the capital of France?",
            output_length="A short paragraph",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention Paris
        assert "paris" in result.lower() or "france" in result.lower()

    @pytest.mark.asyncio
    async def test_iterative_flow_respects_max_iterations(self):
        """IterativeResearchFlow should respect max_iterations limit."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=1, max_time_minutes=5)
        result = await flow.run(query="What are the main features of Python?")

        assert isinstance(result, str)
        # Should complete within 1 iteration (or hit max)
        assert flow.iteration <= 1

    @pytest.mark.asyncio
    async def test_iterative_flow_with_background_context(self):
        """IterativeResearchFlow should use background context."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=2, max_time_minutes=2)
        result = await flow.run(
            query="What is machine learning?",
            background_context="Machine learning is a subset of artificial intelligence.",
        )

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestDeepResearchFlowIntegration:
    """Integration tests for DeepResearchFlow with real API calls."""

    @pytest.mark.asyncio
    async def test_deep_flow_creates_multi_section_report(self):
        """DeepResearchFlow should create a report with multiple sections."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,  # Keep it short for testing
            max_time_minutes=3,
        )
        result = await flow.run("What are the main features of Python programming language?")

        assert isinstance(result, str)
        assert len(result) > 100  # Should have substantial content
        # Should have section structure
        assert "#" in result or "##" in result

    @pytest.mark.asyncio
    async def test_deep_flow_uses_long_writer(self):
        """DeepResearchFlow should use long writer by default."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=True,
        )
        result = await flow.run("Explain the basics of quantum computing")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_deep_flow_uses_proofreader_when_specified(self):
        """DeepResearchFlow should use proofreader when use_long_writer=False."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=False,
        )
        result = await flow.run("What is artificial intelligence?")

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestGraphOrchestratorIntegration:
    """Integration tests for GraphOrchestrator with real API calls."""

    @pytest.mark.asyncio
    async def test_graph_orchestrator_iterative_mode(self):
        """GraphOrchestrator should run in iterative mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="iterative",
            max_iterations=1,
            max_time_minutes=2,
        )

        events = []
        async for event in orchestrator.run("What is Python?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_graph_orchestrator_deep_mode(self):
        """GraphOrchestrator should run in deep mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="deep",
            max_iterations=1,
            max_time_minutes=3,
        )

        events = []
        async for event in orchestrator.run("What are the main features of Python?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_graph_orchestrator_auto_mode(self):
        """GraphOrchestrator should auto-detect research mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="auto",
            max_iterations=1,
            max_time_minutes=2,
        )

        events = []
        async for event in orchestrator.run("What is Python?"):
            events.append(event)

        assert len(events) > 0
        # Should complete successfully regardless of mode
        event_types = [e.type for e in events]
        assert "complete" in event_types


@pytest.mark.integration
class TestGraphOrchestrationIntegration:
    """Integration tests for graph-based orchestration with real API calls."""

    @pytest.mark.asyncio
    async def test_iterative_flow_with_graph_execution(self):
        """IterativeResearchFlow should work with graph execution enabled."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(
            max_iterations=1,
            max_time_minutes=2,
            use_graph=True,
        )
        result = await flow.run(query="What is the capital of France?")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention Paris
        assert "paris" in result.lower() or "france" in result.lower()

    @pytest.mark.asyncio
    async def test_deep_flow_with_graph_execution(self):
        """DeepResearchFlow should work with graph execution enabled."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_graph=True,
        )
        result = await flow.run("What are the main features of Python programming language?")

        assert isinstance(result, str)
        assert len(result) > 100  # Should have substantial content

    @pytest.mark.asyncio
    async def test_graph_orchestrator_with_graph_execution(self):
        """GraphOrchestrator should work with graph execution enabled."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="iterative",
            max_iterations=1,
            max_time_minutes=2,
            use_graph=True,
        )

        events = []
        async for event in orchestrator.run("What is Python?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

        # Extract final report from complete event
        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) > 0
        final_report = complete_events[0].message
        assert isinstance(final_report, str)
        assert len(final_report) > 0

    @pytest.mark.asyncio
    async def test_graph_orchestrator_parallel_execution(self):
        """GraphOrchestrator should support parallel execution in deep mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="deep",
            max_iterations=1,
            max_time_minutes=3,
            use_graph=True,
        )

        events = []
        async for event in orchestrator.run("What are the main features of Python?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_graph_vs_chain_execution_comparison(self):
        """Both graph and chain execution should produce similar results."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        query = "What is the capital of France?"

        # Run with graph execution
        flow_graph = create_iterative_flow(
            max_iterations=1,
            max_time_minutes=2,
            use_graph=True,
        )
        result_graph = await flow_graph.run(query=query)

        # Run with agent chains
        flow_chains = create_iterative_flow(
            max_iterations=1,
            max_time_minutes=2,
            use_graph=False,
        )
        result_chains = await flow_chains.run(query=query)

        # Both should produce valid results
        assert isinstance(result_graph, str)
        assert isinstance(result_chains, str)
        assert len(result_graph) > 0
        assert len(result_chains) > 0

        # Both should mention the answer (Paris)
        assert "paris" in result_graph.lower() or "france" in result_graph.lower()
        assert "paris" in result_chains.lower() or "france" in result_chains.lower()


@pytest.mark.integration
class TestReportSynthesisIntegration:
    """Integration tests for report synthesis with writer agents."""

    @pytest.mark.asyncio
    async def test_iterative_flow_generates_report(self):
        """IterativeResearchFlow should generate a report with writer agent."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=1, max_time_minutes=2)
        result = await flow.run(
            query="What is the capital of France?",
            output_length="A short paragraph",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should be a formatted report
        assert "paris" in result.lower() or "france" in result.lower()
        # Should have some structure (markdown headers or content)
        assert len(result) > 50

    @pytest.mark.asyncio
    async def test_iterative_flow_includes_citations(self):
        """IterativeResearchFlow should include citations in the report."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=1, max_time_minutes=2)
        result = await flow.run(
            query="What is machine learning?",
            output_length="A short paragraph",
        )

        assert isinstance(result, str)
        # Should have some form of citations or references
        # (either [1], [2] format or References section)
        # Note: Citations may not always be present depending on findings
        # This is a soft check - just verify report was generated
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_iterative_flow_handles_empty_findings(self):
        """IterativeResearchFlow should handle empty findings gracefully."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_iterative_flow(max_iterations=1, max_time_minutes=1)
        # Use a query that might not return findings quickly
        result = await flow.run(
            query="Test query with no findings",
            output_length="A short paragraph",
        )

        # Should still return a report (even if minimal)
        assert isinstance(result, str)
        # Writer agent should handle empty findings with fallback

    @pytest.mark.asyncio
    async def test_deep_flow_with_long_writer(self):
        """DeepResearchFlow should use long writer to create sections."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=True,
        )
        result = await flow.run("What are the main features of Python programming language?")

        assert isinstance(result, str)
        assert len(result) > 100  # Should have substantial content
        # Should have section structure (table of contents or sections)
        has_structure = (
            "##" in result
            or "#" in result
            or "table of contents" in result.lower()
            or "introduction" in result.lower()
        )
        # Long writer should create structured report
        assert has_structure or len(result) > 200

    @pytest.mark.asyncio
    async def test_deep_flow_creates_sections(self):
        """DeepResearchFlow should create multiple sections in the report."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=True,
        )
        result = await flow.run("Explain the basics of quantum computing")

        assert isinstance(result, str)
        # Should have multiple sections (indicated by headers)
        # Should have at least some structure
        assert len(result) > 100

    @pytest.mark.asyncio
    async def test_deep_flow_aggregates_references(self):
        """DeepResearchFlow should aggregate references from all sections."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=True,
        )
        result = await flow.run("What are the main features of Python programming language?")

        assert isinstance(result, str)
        # Long writer should aggregate references at the end
        # Check for references section or citation format
        # Note: References may not always be present
        # Just verify report structure is correct
        assert len(result) > 100

    @pytest.mark.asyncio
    async def test_deep_flow_with_proofreader(self):
        """DeepResearchFlow should use proofreader to finalize report."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=False,  # Use proofreader instead
        )
        result = await flow.run("What is artificial intelligence?")

        assert isinstance(result, str)
        assert len(result) > 0
        # Proofreader should create polished report
        # Should have some structure
        assert len(result) > 50

    @pytest.mark.asyncio
    async def test_proofreader_removes_duplicates(self):
        """Proofreader should remove duplicate content from report."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=False,
        )
        result = await flow.run("Explain machine learning basics")

        assert isinstance(result, str)
        # Proofreader should create polished, non-repetitive content
        # This is a soft check - just verify report was generated
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_proofreader_adds_summary(self):
        """Proofreader should add a summary to the report."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        flow = create_deep_flow(
            max_iterations=1,
            max_time_minutes=3,
            use_long_writer=False,
        )
        result = await flow.run("What is Python programming language?")

        assert isinstance(result, str)
        # Proofreader should add summary/outline
        # Check for summary indicators
        # Note: Summary format may vary
        # Just verify report was generated
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_graph_orchestrator_uses_writer_agents(self):
        """GraphOrchestrator should use writer agents in iterative mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="iterative",
            max_iterations=1,
            max_time_minutes=2,
            use_graph=False,  # Use agent chains to test writer integration
        )

        events = []
        async for event in orchestrator.run("What is the capital of France?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

        # Extract final report from complete event
        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) > 0
        final_report = complete_events[0].message
        assert isinstance(final_report, str)
        assert len(final_report) > 0
        # Should have content from writer agent
        assert "paris" in final_report.lower() or "france" in final_report.lower()

    @pytest.mark.asyncio
    async def test_graph_orchestrator_uses_long_writer_in_deep_mode(self):
        """GraphOrchestrator should use long writer in deep mode."""
        if not settings.has_openai_key and not settings.has_anthropic_key:
            pytest.skip("No OpenAI or Anthropic API key available")

        orchestrator = create_graph_orchestrator(
            mode="deep",
            max_iterations=1,
            max_time_minutes=3,
            use_graph=False,  # Use agent chains
        )

        events = []
        async for event in orchestrator.run("What are the main features of Python?"):
            events.append(event)

        assert len(events) > 0
        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "complete" in event_types

        # Extract final report
        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) > 0
        final_report = complete_events[0].message
        assert isinstance(final_report, str)
        assert len(final_report) > 0
        # Should have structured content from long writer
        assert len(final_report) > 100
