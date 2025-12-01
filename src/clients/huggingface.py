"""HuggingFace Chat Client adapter for Microsoft Agent Framework.

This client enables the use of HuggingFace Inference API (including the free tier)
as a backend for the agent framework, allowing "Advanced Mode" to work without
an OpenAI API key.
"""

import asyncio
from collections.abc import AsyncIterable, MutableSequence
from functools import partial
from typing import Any, cast

import structlog
from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
)
from agent_framework._types import FunctionCallContent
from huggingface_hub import InferenceClient

from src.utils.config import settings

logger = structlog.get_logger()


class HuggingFaceChatClient(BaseChatClient):  # type: ignore[misc]
    """Adapter for HuggingFace Inference API with full function calling support."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HuggingFace chat client.

        Args:
            model_id: The HuggingFace model ID (default: configured value or Qwen2.5-72B).
            api_key: HF_TOKEN (optional, defaults to env var).
            **kwargs: Additional arguments passed to BaseChatClient.
        """
        super().__init__(**kwargs)
        self.model_id = model_id or settings.huggingface_model or "Qwen/Qwen2.5-72B-Instruct"
        self.api_key = api_key or settings.hf_token

        # Initialize the HF Inference Client
        # timeout=60 to prevent premature timeouts on long reasonings
        self._client = InferenceClient(
            model=self.model_id,
            token=self.api_key,
            timeout=60,
        )
        logger.info("Initialized HuggingFaceChatClient", model=self.model_id)

    def _convert_messages(self, messages: MutableSequence[ChatMessage]) -> list[dict[str, Any]]:
        """Convert framework messages to HuggingFace format."""
        hf_messages: list[dict[str, Any]] = []
        for msg in messages:
            # Basic conversion - extend as needed for multi-modal
            content = msg.text or ""
            # msg.role can be string or enum - extract .value for enums
            # str(Role.USER) -> "Role.USER" (wrong), Role.USER.value -> "user" (correct)
            if hasattr(msg.role, "value"):
                role_str = str(msg.role.value)
            else:
                role_str = str(msg.role)
            hf_messages.append({"role": role_str, "content": content})
        return hf_messages

    def _convert_tools(self, tools: list[Any] | None) -> list[dict[str, Any]] | None:
        """Convert AIFunction objects to OpenAI-compatible tool definitions.

        AIFunction.to_dict() returns:
            {'type': 'ai_function', 'name': '...', 'input_model': {...}}

        OpenAI/HuggingFace expects:
            {'type': 'function', 'function': {'name': '...', 'parameters': {...}}}
        """
        if not tools:
            return None

        json_tools = []
        for tool in tools:
            if hasattr(tool, "to_dict"):
                try:
                    t_dict = tool.to_dict()
                    json_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t_dict["name"],
                                "description": t_dict.get("description", ""),
                                "parameters": t_dict["input_model"],
                            },
                        }
                    )
                except (KeyError, TypeError) as e:
                    logger.warning("Failed to convert tool", tool=str(tool), error=str(e))
            elif isinstance(tool, dict):
                # Already a dict - assume correct format
                json_tools.append(tool)
            else:
                logger.warning("Skipping non-serializable tool", tool_type=str(type(tool)))

        return json_tools if json_tools else None

    def _parse_tool_calls(self, message: Any) -> list[FunctionCallContent]:
        """Parse HuggingFace tool_calls into framework FunctionCallContent.

        HF returns tool_calls as:
            [ChatCompletionOutputToolCall(id='...', function=ChatCompletionOutputFunctionDefinition(
                name='...', arguments='{"key": "value"}'), type='function')]
        """
        contents: list[FunctionCallContent] = []

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return contents

        for tc in message.tool_calls:
            try:
                contents.append(
                    FunctionCallContent(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,  # JSON string or dict
                    )
                )
            except (AttributeError, TypeError) as e:
                logger.warning("Failed to parse tool call", error=str(e))

        return contents

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """Synchronous response generation using chat_completion."""
        hf_messages = self._convert_messages(messages)

        # Convert AIFunction objects to OpenAI-compatible JSON
        tools = self._convert_tools(chat_options.tools if chat_options.tools else None)

        # HF expects 'tool_choice' to be 'auto', 'none', or specific tool
        # Framework uses ToolMode enum or dict
        hf_tool_choice: str | None = None
        if tools and chat_options.tool_choice is not None:
            tool_choice_str = str(chat_options.tool_choice)
            if "AUTO" in tool_choice_str:
                hf_tool_choice = "auto"
            # For NONE or other, leave as None

        try:
            # Use explicit None checks - 'or' treats 0/0.0 as falsy
            # temperature=0.0 is valid (deterministic output)
            max_tokens = chat_options.max_tokens if chat_options.max_tokens is not None else 2048
            temperature = chat_options.temperature if chat_options.temperature is not None else 0.7

            # Use partial to create a callable with keyword args for to_thread
            call_fn = partial(
                self._client.chat_completion,
                messages=hf_messages,
                tools=tools,
                tool_choice=hf_tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

            response = await asyncio.to_thread(call_fn)

            # Parse response
            # HF returns a ChatCompletionOutput
            choices = response.choices
            if not choices:
                return ChatResponse(messages=[], response_id="error-no-choices")

            choice = choices[0]
            message = choice.message
            message_content = message.content or ""

            # Parse tool calls if present
            tool_call_contents = self._parse_tool_calls(message)

            # Construct response message with tool calls in contents
            response_msg = ChatMessage(
                role=cast(Any, message.role),
                text=message_content,
                contents=tool_call_contents if tool_call_contents else None,
            )

            return ChatResponse(
                messages=[response_msg],
                response_id=response.id or "hf-response",
            )

        except Exception as e:
            logger.error("HuggingFace API error", error=str(e))
            raise

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """Streaming response generation."""
        hf_messages = self._convert_messages(messages)

        # Convert AIFunction objects to OpenAI-compatible JSON
        tools = self._convert_tools(chat_options.tools if chat_options.tools else None)

        hf_tool_choice: str | None = None
        if tools and chat_options.tool_choice is not None:
            if "AUTO" in str(chat_options.tool_choice):
                hf_tool_choice = "auto"

        try:
            # Use explicit None checks - 'or' treats 0/0.0 as falsy
            # temperature=0.0 is valid (deterministic output)
            max_tokens = chat_options.max_tokens if chat_options.max_tokens is not None else 2048
            temperature = chat_options.temperature if chat_options.temperature is not None else 0.7

            # Use partial for streaming call
            call_fn = partial(
                self._client.chat_completion,
                messages=hf_messages,
                tools=tools,
                tool_choice=hf_tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            stream = await asyncio.to_thread(call_fn)

            for chunk in stream:
                # Chunk is ChatCompletionStreamOutput
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Convert to ChatResponseUpdate
                yield ChatResponseUpdate(
                    role=cast(Any, delta.role) if delta.role else None,
                    content=delta.content,
                )

                # Yield control to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("HuggingFace Streaming error", error=str(e))
            raise
