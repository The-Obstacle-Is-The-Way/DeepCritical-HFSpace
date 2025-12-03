"""HuggingFace Chat Client adapter for Microsoft Agent Framework.

This client enables the use of HuggingFace Inference API (including the free tier)
as a backend for the agent framework, allowing "Advanced Mode" to work without
an OpenAI API key.
"""

import asyncio
import json
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
    FinishReason,
    Role,
)
from agent_framework._middleware import use_chat_middleware
from agent_framework._tools import use_function_invocation
from agent_framework._types import FunctionCallContent, FunctionResultContent
from agent_framework.observability import use_observability
from huggingface_hub import InferenceClient

from src.utils.config import settings

logger = structlog.get_logger()


@use_function_invocation
@use_observability
@use_chat_middleware
class HuggingFaceChatClient(BaseChatClient):  # type: ignore[misc]
    """Adapter for HuggingFace Inference API with full function calling support."""

    # Marker to tell agent_framework that this client supports function calling
    # Without this, the framework warns and ignores tools
    __function_invoking_chat_client__ = True

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HuggingFace chat client.

        Args:
            model_id: The HuggingFace model ID (default: configured value or Qwen2.5-7B).
            api_key: HF_TOKEN (optional, defaults to env var).
            **kwargs: Additional arguments passed to BaseChatClient.
        """
        super().__init__(**kwargs)
        # FIX: Use 7B model to stay on HuggingFace native infrastructure (avoid Novita 500s)
        self.model_id = model_id or settings.huggingface_model or "Qwen/Qwen2.5-7B-Instruct"
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

        # Track call_id -> tool_name mapping for tool result messages
        # Assistant messages with tool_calls come before tool result messages
        call_id_to_name: dict[str, str] = {}

        for msg in messages:
            # msg.role can be string or enum - extract .value for enums
            if hasattr(msg.role, "value"):
                role_str = str(msg.role.value)
            else:
                role_str = str(msg.role)

            content_str = msg.text or ""
            tool_calls = []
            tool_call_id = None
            tool_name = None

            # Process contents for tool calls and results
            if msg.contents:
                for item in msg.contents:
                    if isinstance(item, FunctionCallContent):
                        # This is an assistant message invoking a tool
                        # Track call_id -> name for later tool result messages
                        call_id_to_name[item.call_id] = item.name
                        tool_calls.append(
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": (
                                        item.arguments
                                        if isinstance(item.arguments, str)
                                        else json.dumps(item.arguments)
                                    ),
                                },
                            }
                        )
                    elif isinstance(item, FunctionResultContent):
                        # This is a tool result message
                        role_str = "tool"
                        tool_call_id = item.call_id
                        # Look up tool name from prior FunctionCallContent
                        tool_name = call_id_to_name.get(item.call_id)
                        # For tool results, JSON-encode structured data
                        # HuggingFace/OpenAI expects string content
                        if item.result is None:
                            content_str = ""
                        elif isinstance(item.result, str):
                            content_str = item.result
                        else:
                            content_str = json.dumps(item.result)

            message_dict: dict[str, Any] = {"role": role_str, "content": content_str}

            if tool_calls:
                message_dict["tool_calls"] = tool_calls

            if tool_call_id:
                message_dict["tool_call_id"] = tool_call_id
                # Add name field if we tracked it (required by some APIs)
                if tool_name:
                    message_dict["name"] = tool_name

            hf_messages.append(message_dict)

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
        """Parse HuggingFace tool_calls into framework FunctionCallContent."""
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

            # Accumulator for tool calls (index -> dict)
            # We need to accumulate because deltas are partial
            tool_call_accumulator: dict[int, dict[str, Any]] = {}

            for chunk in stream:
                # Chunk is ChatCompletionStreamOutput
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # 1. Handle Text Content
                if delta.content:
                    yield ChatResponseUpdate(
                        role=cast(Any, delta.role) if delta.role else None,
                        text=delta.content,
                    )

                # 2. Handle Tool Calls (Accumulate)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }

                        # Merge fields
                        if tc.id:
                            tool_call_accumulator[idx]["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_call_accumulator[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_call_accumulator[idx]["arguments"] += tc.function.arguments

                # Yield control to event loop
                await asyncio.sleep(0)

            # 3. Yield Accumulated Tool Calls
            if tool_call_accumulator:
                contents: list[FunctionCallContent] = []
                for idx in sorted(tool_call_accumulator.keys()):
                    tc_data = tool_call_accumulator[idx]
                    # Only yield if ID and Name are present (required by FunctionCallContent)
                    if tc_data["id"] and tc_data["name"]:
                        contents.append(
                            FunctionCallContent(
                                call_id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=tc_data["arguments"],
                            )
                        )

                if contents:
                    yield ChatResponseUpdate(
                        contents=contents,
                        role=Role.ASSISTANT,
                        finish_reason=FinishReason.TOOL_CALLS,
                    )

        except Exception as e:
            logger.error("HuggingFace Streaming error", error=str(e))
            raise
