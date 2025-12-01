"""Base classes for Chat Client implementations.

This module re-exports the BaseChatClient and related types from the core
agent_framework package to provide a single point of import for the project.
"""

from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
)

__all__ = [
    "BaseChatClient",
    "ChatMessage",
    "ChatResponse",
    "ChatResponseUpdate",
]
