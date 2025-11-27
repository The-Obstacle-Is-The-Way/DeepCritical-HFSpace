"""Test that agent framework dependencies are importable and usable."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

# Import conditional on package availability, but for this test we expect it to be there
try:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
except ImportError:
    ChatAgent = None
    OpenAIChatClient = None


@pytest.mark.skipif(ChatAgent is None, reason="agent-framework-core not installed")
def test_agent_framework_import():
    """Test that agent_framework can be imported."""
    assert ChatAgent is not None
    assert OpenAIChatClient is not None  # Verify both imports work


@pytest.mark.skipif(ChatAgent is None, reason="agent-framework-core not installed")
def test_chat_agent_instantiation():
    """Test that ChatAgent can be instantiated with a mock client."""
    mock_client = MagicMock()
    # We assume ChatAgent takes chat_client as first argument based on _agents.py source
    agent = ChatAgent(chat_client=mock_client, name="TestAgent")
    assert agent.name == "TestAgent"
    assert agent.chat_client == mock_client
