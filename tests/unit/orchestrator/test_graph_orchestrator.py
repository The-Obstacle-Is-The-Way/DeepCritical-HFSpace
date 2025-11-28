"""Unit tests for GraphOrchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.graph_orchestrator import (
    GraphExecutionContext,
    GraphOrchestrator,
    create_graph_orchestrator,
)
from src.utils.models import AgentEvent


class TestGraphExecutionContext:
    """Tests for GraphExecutionContext."""

    def test_context_initialization(self):
        """Test initializing execution context."""
        from src.middleware.budget_tracker import BudgetTracker
        from src.middleware.state_machine import WorkflowState

        state = WorkflowState()
        budget_tracker = BudgetTracker()
        context = GraphExecutionContext(state, budget_tracker)

        assert context.state == state
        assert context.budget_tracker == budget_tracker
        assert context.current_node == ""
        assert len(context.visited_nodes) == 0
        assert len(context.node_results) == 0

    def test_set_and_get_node_result(self):
        """Test setting and getting node results."""
        from src.middleware.budget_tracker import BudgetTracker
        from src.middleware.state_machine import WorkflowState

        context = GraphExecutionContext(WorkflowState(), BudgetTracker())
        context.set_node_result("node1", "result1")
        assert context.get_node_result("node1") == "result1"
        assert context.get_node_result("nonexistent") is None

    def test_visited_nodes_tracking(self):
        """Test visited nodes tracking."""
        from src.middleware.budget_tracker import BudgetTracker
        from src.middleware.state_machine import WorkflowState

        context = GraphExecutionContext(WorkflowState(), BudgetTracker())
        assert not context.has_visited("node1")
        context.mark_visited("node1")
        assert context.has_visited("node1")


class TestGraphOrchestrator:
    """Tests for GraphOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a GraphOrchestrator instance."""
        return GraphOrchestrator(
            mode="iterative",
            max_iterations=5,
            max_time_minutes=10,
            use_graph=True,
        )

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = GraphOrchestrator(
            mode="iterative",
            max_iterations=3,
            max_time_minutes=5,
            use_graph=True,
        )
        assert orchestrator.mode == "iterative"
        assert orchestrator.max_iterations == 3
        assert orchestrator.max_time_minutes == 5
        assert orchestrator.use_graph is True

    def test_initialization_with_agent_chains(self):
        """Test orchestrator initialization with agent chains."""
        orchestrator = GraphOrchestrator(
            mode="deep",
            max_iterations=5,
            max_time_minutes=10,
            use_graph=False,
        )
        assert orchestrator.use_graph is False
        assert orchestrator._iterative_flow is None
        assert orchestrator._deep_flow is None

    @pytest.mark.asyncio
    async def test_detect_research_mode_deep(self):
        """Test detecting deep research mode from query."""
        orchestrator = GraphOrchestrator(mode="auto")
        mode = await orchestrator._detect_research_mode("Create a report with sections about X")
        assert mode == "deep"

    @pytest.mark.asyncio
    async def test_detect_research_mode_iterative(self):
        """Test detecting iterative research mode from query."""
        orchestrator = GraphOrchestrator(mode="auto")
        mode = await orchestrator._detect_research_mode("What is the mechanism of action?")
        assert mode == "iterative"

    @pytest.mark.asyncio
    async def test_run_with_chains_iterative(self):
        """Test running with agent chains (iterative mode)."""
        orchestrator = GraphOrchestrator(
            mode="iterative",
            max_iterations=2,
            max_time_minutes=5,
            use_graph=False,
        )

        # Mock the flow class
        with patch("src.orchestrator.graph_orchestrator.IterativeResearchFlow") as mock_flow_class:
            mock_flow = AsyncMock()
            mock_flow.run = AsyncMock(return_value="# Report\n\nContent")
            mock_flow_class.return_value = mock_flow

            events = []
            async for event in orchestrator.run("Test query"):
                events.append(event)

            assert len(events) >= 2  # Should have started and complete events
            assert events[0].type == "started"
            complete_events = [e for e in events if e.type == "complete"]
            assert len(complete_events) > 0

    @pytest.mark.asyncio
    async def test_run_with_chains_deep(self):
        """Test running with agent chains (deep mode)."""
        orchestrator = GraphOrchestrator(
            mode="deep",
            max_iterations=2,
            max_time_minutes=5,
            use_graph=False,
        )

        # Mock the flow class
        with patch("src.orchestrator.graph_orchestrator.DeepResearchFlow") as mock_flow_class:
            mock_flow = AsyncMock()
            mock_flow.run = AsyncMock(return_value="# Report\n\nContent")
            mock_flow_class.return_value = mock_flow

            events = []
            async for event in orchestrator.run("Test query"):
                events.append(event)

            assert len(events) >= 2
            assert events[0].type == "started"
            complete_events = [e for e in events if e.type == "complete"]
            assert len(complete_events) > 0

    @pytest.mark.asyncio
    async def test_run_with_graph_iterative(self):
        """Test running with graph execution (iterative mode)."""
        orchestrator = GraphOrchestrator(
            mode="iterative",
            max_iterations=2,
            max_time_minutes=5,
            use_graph=True,
        )

        # Mock the graph building by patching the _build_graph method
        async def mock_build_graph(mode: str):
            from src.agent_factory.graph_builder import ResearchGraph

            mock_graph = MagicMock(spec=ResearchGraph)
            mock_graph.entry_node = "start"
            mock_graph.exit_nodes = ["end"]
            mock_graph.get_node = MagicMock(return_value=None)
            mock_graph.get_next_nodes = MagicMock(return_value=[])
            return mock_graph

        orchestrator._build_graph = mock_build_graph

        # Mock the graph execution
        async def mock_run_with_graph(query: str, mode: str):
            yield AgentEvent(type="started", message="Starting", iteration=0)
            yield AgentEvent(type="looping", message="Processing", iteration=1)
            yield AgentEvent(type="complete", message="# Final Report\n\nContent", iteration=1)

        orchestrator._run_with_graph = mock_run_with_graph

        events = []
        async for event in orchestrator.run("Test query"):
            events.append(event)

        # Should have events from graph execution
        assert len(events) > 0
        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) > 0

    @pytest.mark.asyncio
    async def test_run_handles_errors(self):
        """Test that run handles errors gracefully."""
        orchestrator = GraphOrchestrator(
            mode="iterative",
            max_iterations=2,
            max_time_minutes=5,
            use_graph=False,
        )
        # Ensure flow is None so it gets created fresh
        orchestrator._iterative_flow = None

        # Create the flow first, then patch its run method
        from src.orchestrator.research_flow import IterativeResearchFlow

        # Create flow and patch its run method to raise exception
        original_flow = IterativeResearchFlow(
            max_iterations=2,
            max_time_minutes=5,
        )
        orchestrator._iterative_flow = original_flow

        with patch.object(original_flow, "run", side_effect=Exception("Test error")):
            events = []
            # Collect events manually to ensure we get error events even when exception occurs
            gen = orchestrator.run("Test query")
            while True:
                try:
                    event = await gen.__anext__()
                    events.append(event)
                    # If we got an error event, continue to see if outer handler also yields one
                    if event.type == "error":
                        # Try to get outer handler's error event too
                        try:
                            next_event = await gen.__anext__()
                            events.append(next_event)
                        except (StopAsyncIteration, Exception):
                            break
                        break
                except StopAsyncIteration:
                    break
                except Exception:
                    # Exception occurred - outer handler should yield error event
                    # Try to get it
                    try:
                        event = await gen.__anext__()
                        events.append(event)
                    except (StopAsyncIteration, Exception):
                        break
                    break

            error_events = [e for e in events if e.type == "error"]
            assert len(error_events) > 0, (
                f"No error events found. Events: {[e.type for e in events]}"
            )
            assert (
                "error" in error_events[0].message.lower()
                or "failed" in error_events[0].message.lower()
            )

    @pytest.mark.asyncio
    async def test_build_graph_iterative(self):
        """Test building iterative graph."""
        orchestrator = GraphOrchestrator(mode="iterative", use_graph=True)

        with (
            patch("src.orchestrator.graph_orchestrator.create_knowledge_gap_agent") as mock_kg,
            patch("src.orchestrator.graph_orchestrator.create_tool_selector_agent") as mock_ts,
            patch("src.orchestrator.graph_orchestrator.create_thinking_agent") as mock_thinking,
            patch("src.orchestrator.graph_orchestrator.create_writer_agent") as mock_writer,
            patch(
                "src.orchestrator.graph_orchestrator.create_iterative_graph"
            ) as mock_create_graph,
        ):
            mock_kg_agent = MagicMock()
            mock_kg_agent.agent = MagicMock()
            mock_kg.return_value = mock_kg_agent

            mock_ts_agent = MagicMock()
            mock_ts_agent.agent = MagicMock()
            mock_ts.return_value = mock_ts_agent

            mock_thinking_agent = MagicMock()
            mock_thinking_agent.agent = MagicMock()
            mock_thinking.return_value = mock_thinking_agent

            mock_writer_agent = MagicMock()
            mock_writer_agent.agent = MagicMock()
            mock_writer.return_value = mock_writer_agent

            mock_graph = MagicMock()
            mock_create_graph.return_value = mock_graph

            graph = await orchestrator._build_graph("iterative")
            assert graph == mock_graph

    @pytest.mark.asyncio
    async def test_build_graph_deep(self):
        """Test building deep graph."""
        orchestrator = GraphOrchestrator(mode="deep", use_graph=True)

        with (
            patch("src.orchestrator.graph_orchestrator.create_planner_agent") as mock_planner,
            patch("src.orchestrator.graph_orchestrator.create_knowledge_gap_agent") as mock_kg,
            patch("src.orchestrator.graph_orchestrator.create_tool_selector_agent") as mock_ts,
            patch("src.orchestrator.graph_orchestrator.create_thinking_agent") as mock_thinking,
            patch("src.orchestrator.graph_orchestrator.create_writer_agent") as mock_writer,
            patch(
                "src.orchestrator.graph_orchestrator.create_long_writer_agent"
            ) as mock_long_writer,
            patch("src.orchestrator.graph_orchestrator.create_deep_graph") as mock_create_graph,
        ):
            # Setup all mocks
            for mock_func in [
                mock_planner,
                mock_kg,
                mock_ts,
                mock_thinking,
                mock_writer,
                mock_long_writer,
            ]:
                mock_agent = MagicMock()
                mock_agent.agent = MagicMock()
                mock_func.return_value = mock_agent

            mock_graph = MagicMock()
            mock_create_graph.return_value = mock_graph

            graph = await orchestrator._build_graph("deep")
            assert graph == mock_graph


class TestCreateGraphOrchestrator:
    """Tests for create_graph_orchestrator factory function."""

    def test_create_with_defaults(self):
        """Test creating orchestrator with defaults."""
        orchestrator = create_graph_orchestrator()
        assert isinstance(orchestrator, GraphOrchestrator)
        assert orchestrator.mode == "auto"
        assert orchestrator.max_iterations == 5
        assert orchestrator.max_time_minutes == 10

    def test_create_with_custom_params(self):
        """Test creating orchestrator with custom parameters."""
        orchestrator = create_graph_orchestrator(
            mode="deep",
            max_iterations=10,
            max_time_minutes=20,
            use_graph=False,
        )
        assert orchestrator.mode == "deep"
        assert orchestrator.max_iterations == 10
        assert orchestrator.max_time_minutes == 20
        assert orchestrator.use_graph is False
