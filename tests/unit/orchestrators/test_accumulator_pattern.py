"""
Test the Accumulator Pattern for Microsoft Agent Framework event handling.

This tests SPEC 17: We use MagenticAgentDeltaEvent.text as the sole source of content,
and MagenticAgentMessageEvent as a signal only (ignoring .message to avoid repr bug).
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# --- Create real event classes ---
class MockDeltaEvent:
    """Simulates MagenticAgentDeltaEvent with streaming text."""

    def __init__(self, text: str, agent_id: str = "TestAgent"):
        self.text = text
        self.agent_id = agent_id


class MockMessageEvent:
    """Simulates MagenticAgentMessageEvent with potentially corrupted .message."""

    def __init__(self, message_text: str, agent_id: str = "TestAgent"):
        self.message = MagicMock()
        self.message.text = message_text  # This could be repr garbage
        self.agent_id = agent_id
        self.text = None  # No top-level .text on MessageEvent


class MockFinalResultEvent:
    """Simulates MagenticFinalResultEvent."""

    def __init__(self, text: str):
        self.message = MagicMock()
        self.message.text = text
        self.text = None


class MockOrchestratorMessageEvent:
    """Simulates MagenticOrchestratorMessageEvent."""

    def __init__(self, kind: str = "user_task", message: str = "test"):
        self.kind = kind
        self.message = MagicMock()
        self.message.text = message


class MockWorkflowOutputEvent:
    """Simulates WorkflowOutputEvent."""

    def __init__(self, data=None):
        self.data = data


# Pass-through decorators
def mock_use_function_invocation(func=None):
    return func if func else lambda f: f


def mock_use_observability(func=None):
    return func if func else lambda f: f


@pytest.fixture
def mock_agent_framework():
    """
    Mock the agent_framework module structure in sys.modules.
    """
    # Create the mock module structure
    mock_af = types.ModuleType("agent_framework")
    mock_af_openai = types.ModuleType("agent_framework.openai")
    mock_af_middleware = types.ModuleType("agent_framework._middleware")
    mock_af_tools = types.ModuleType("agent_framework._tools")
    mock_af_types = types.ModuleType("agent_framework._types")
    mock_af_observability = types.ModuleType("agent_framework.observability")

    # Populate submodules
    mock_af.openai = mock_af_openai
    mock_af._middleware = mock_af_middleware
    mock_af._tools = mock_af_tools
    mock_af._types = mock_af_types
    mock_af.observability = mock_af_observability

    # Assign our REAL event classes as the module-level types
    mock_af.MagenticAgentDeltaEvent = MockDeltaEvent
    mock_af.MagenticAgentMessageEvent = MockMessageEvent
    mock_af.MagenticFinalResultEvent = MockFinalResultEvent
    mock_af.MagenticOrchestratorMessageEvent = MockOrchestratorMessageEvent
    mock_af.WorkflowOutputEvent = MockWorkflowOutputEvent

    # Mock other classes
    mock_af.MagenticBuilder = MagicMock
    mock_af.ChatAgent = MagicMock
    mock_af.ai_function = MagicMock
    mock_af.BaseChatClient = MagicMock
    mock_af.ToolProtocol = MagicMock
    mock_af.ChatMessage = MagicMock
    mock_af.ChatResponse = MagicMock
    mock_af.ChatResponseUpdate = MagicMock
    mock_af.ChatOptions = MagicMock
    mock_af.FinishReason = MagicMock
    mock_af.Role = MagicMock

    # Populate symbols in submodules
    mock_af_openai.OpenAIChatClient = MagicMock
    mock_af_middleware.use_chat_middleware = MagicMock
    mock_af_tools.use_function_invocation = mock_use_function_invocation
    mock_af_types.FunctionCallContent = MagicMock
    mock_af_types.FunctionResultContent = MagicMock
    mock_af_observability.use_observability = mock_use_observability

    # Patch sys.modules to include our mocks
    with patch.dict(
        sys.modules,
        {
            "agent_framework": mock_af,
            "agent_framework.openai": mock_af_openai,
            "agent_framework._middleware": mock_af_middleware,
            "agent_framework._tools": mock_af_tools,
            "agent_framework._types": mock_af_types,
            "agent_framework.observability": mock_af_observability,
        },
    ):
        yield mock_af


@pytest.fixture(scope="module", autouse=True)
def cleanup_orchestrator_module():
    """
    Ensure src.orchestrators.advanced is restored to a clean state after tests.
    This prevents 'Mock' classes from leaking into other tests via module globals.
    """
    yield
    # After all tests in this module, reload the orchestrator module
    # This will use the REAL agent_framework (since the mock fixture is teardown)
    import src.orchestrators.advanced

    importlib.reload(src.orchestrators.advanced)


@pytest.fixture
def mock_orchestrator(mock_agent_framework):
    """
    Create an AdvancedOrchestrator with all dependencies mocked.
    Relies on reloading the module to pick up the mocked agent_framework.
    """
    # Import locally
    import src.orchestrators.advanced

    # RELOAD to ensure it picks up the mocked agent_framework from sys.modules
    importlib.reload(src.orchestrators.advanced)

    from src.orchestrators.advanced import AdvancedOrchestrator

    with (
        patch("src.orchestrators.advanced.get_chat_client"),
        patch("src.orchestrators.advanced.get_embedding_service_if_available", return_value=None),
        patch("src.orchestrators.advanced.init_magentic_state"),
        patch("src.agents.state.ResearchMemory"),
        patch("src.utils.service_loader.get_embedding_service", return_value=MagicMock()),
    ):
        orch = AdvancedOrchestrator(max_rounds=5)
        yield orch


@pytest.mark.unit
@pytest.mark.asyncio
async def test_accumulator_pattern_scenario_a_standard_text(mock_orchestrator):
    """
    Scenario A: Standard Text Message
    Input: Deltas ("Hello", " World") -> MessageEvent (Poisoned Repr)
    Expected: AgentEvent with "Hello World", NOT the repr string
    """
    events = [
        MockDeltaEvent("Hello", agent_id="ChatBot"),
        MockDeltaEvent(" World", agent_id="ChatBot"),
        MockMessageEvent("<ChatMessage object at 0xDEADBEEF>", agent_id="ChatBot"),
    ]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    mock_workflow = MagicMock()
    mock_workflow.run_stream = mock_stream

    with patch.object(mock_orchestrator, "_build_workflow", return_value=mock_workflow):
        generated_events = []
        async for event in mock_orchestrator.run("test query"):
            generated_events.append(event)

    # Find the completion event for ChatBot (non-streaming)
    chat_events = [
        e for e in generated_events if "ChatBot" in str(e.message) and e.type != "streaming"
    ]

    assert len(chat_events) >= 1, (
        f"Expected ChatBot events, got: {[e.message for e in generated_events]}"
    )
    final_event = chat_events[0]

    # CRITICAL: Must contain accumulated text, NOT repr
    assert "Hello World" in final_event.message or "Hello" in final_event.message
    assert "<ChatMessage" not in final_event.message, f"Repr bug! Got: {final_event.message}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_accumulator_pattern_scenario_b_tool_call(mock_orchestrator):
    """
    Scenario B: Tool Call (No Text Deltas)
    Input: No Deltas -> MessageEvent (Poisoned Repr)
    Expected: AgentEvent with fallback text, NOT the repr string
    """
    events = [
        MockMessageEvent("<ChatMessage object at 0xDEADBEEF>", agent_id="SearchAgent"),
    ]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    mock_workflow = MagicMock()
    mock_workflow.run_stream = mock_stream

    with patch.object(mock_orchestrator, "_build_workflow", return_value=mock_workflow):
        generated_events = []
        async for event in mock_orchestrator.run("test query"):
            generated_events.append(event)

    # Find completion events for SearchAgent
    search_events = [
        e for e in generated_events if "SearchAgent" in str(e.message) and e.type != "streaming"
    ]

    assert len(search_events) >= 1, (
        f"Expected SearchAgent events, got: {[e.message for e in generated_events]}"
    )
    final_event = search_events[0]

    # CRITICAL: Should use fallback, NOT repr
    assert "<ChatMessage" not in final_event.message, f"Repr bug! Got: {final_event.message}"
    # Should contain fallback or tool indicator
    assert "Action completed" in final_event.message or "Tool" in final_event.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_accumulator_pattern_buffer_clearing(mock_orchestrator):
    """
    Verify buffer clears between agents.
    Agent B should NOT inherit Agent A's accumulated text.
    """
    events = [
        MockDeltaEvent("Agent A says hi", agent_id="AgentA"),
        MockMessageEvent("irrelevant", agent_id="AgentA"),
        MockDeltaEvent("Agent B responds", agent_id="AgentB"),
        MockMessageEvent("irrelevant", agent_id="AgentB"),
    ]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    mock_workflow = MagicMock()
    mock_workflow.run_stream = mock_stream

    with patch.object(mock_orchestrator, "_build_workflow", return_value=mock_workflow):
        generated_events = []
        async for event in mock_orchestrator.run("test query"):
            generated_events.append(event)

    # Find non-streaming events for each agent
    agent_a_events = [
        e for e in generated_events if "AgentA" in str(e.message) and e.type != "streaming"
    ]
    agent_b_events = [
        e for e in generated_events if "AgentB" in str(e.message) and e.type != "streaming"
    ]

    # Both should have completion events
    assert len(agent_a_events) >= 1, f"No AgentA events: {[e.message for e in generated_events]}"
    assert len(agent_b_events) >= 1, f"No AgentB events: {[e.message for e in generated_events]}"

    # Agent A should have its own text
    assert "Agent A" in agent_a_events[0].message
    # Agent B should have its own text, NOT Agent A's
    assert "Agent B" in agent_b_events[0].message
    assert "Agent A" not in agent_b_events[0].message, "Buffer not cleared between agents!"
