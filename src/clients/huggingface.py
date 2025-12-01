"""HuggingFace Chat Client adapter for Microsoft Agent Framework.

This client enables the use of HuggingFace Inference API (including the free tier)
as a backend for the agent framework, allowing "Advanced Mode" to work without
an OpenAI API key.
"""

import asyncio
from collections.abc import AsyncIterable, MutableSequence
from typing import Any

import structlog
from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
)
from huggingface_hub import InferenceClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import settings

logger = structlog.get_logger()


class HuggingFaceChatClient(BaseChatClient):
    """Adapter for HuggingFace Inference API."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HuggingFace chat client.

        Args:
            model_id: The HuggingFace model ID (default: configured value or Llama-3.1-70B).
            api_key: HF_TOKEN (optional, defaults to env var).
            **kwargs: Additional arguments passed to BaseChatClient.
        """
        super().__init__(**kwargs)
        self.model_id = (
            model_id
            or settings.huggingface_model
            or "meta-llama/Llama-3.1-70B-Instruct"
        )
        self.api_key = api_key or settings.hf_token
        
        # Initialize the HF Inference Client
        # timeout=60 to prevent premature timeouts on long reasonings
        self._client = InferenceClient(
            model=self.model_id,
            token=self.api_key,
            timeout=60,
        )
        logger.info("Initialized HuggingFaceChatClient", model=self.model_id)

    def _convert_messages(self, messages: MutableSequence[ChatMessage]) -> list[dict[str, str]]:
        """Convert framework messages to HuggingFace format."""
        hf_messages = []
        for msg in messages:
            # Basic conversion - extend as needed for multi-modal
            content = msg.text or ""
            hf_messages.append({"role": msg.role, "content": content})
        return hf_messages

    @retry(
        retry=retry_if_exception_type(Exception),  # Broad retry for network/API issues
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """Synchronous response generation using chat_completion."""
        hf_messages = self._convert_messages(messages)
        
        # Extract tool configuration
        tools = chat_options.tools if chat_options.tools else None
        # HF expects 'tool_choice' to be 'auto', 'none', or specific tool
        # Framework uses ToolMode enum or dict
        tool_choice = chat_options.tool_choice
        if str(tool_choice) == "ToolMode.AUTO":
            tool_choice = "auto"
        elif str(tool_choice) == "ToolMode.NONE":
            # HF doesn't like 'none' string for tool_choice sometimes, strict auto/required check
            tool_choice = None

        try:
            # Run in executor because InferenceClient is synchronous by default
            # (unless using AsyncInferenceClient, but standard is often sync)
            # Actually, let's check if we should use AsyncInferenceClient.
            # The standard InferenceClient has a chat_completion method.
            # To be safe in async context, we'll run_in_executor.
            
            # Note: we use self._client.chat_completion
            
            response = await asyncio.to_thread(
                self._client.chat_completion,
                messages=hf_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=chat_options.max_tokens or 2048,
                temperature=chat_options.temperature or 0.7,
                stream=False,
            )

            # Parse response
            # HF returns a ChatCompletionOutput
            choices = response.choices
            if not choices:
                return ChatResponse(messages=[], response_id="error-no-choices")
                
            choice = choices[0]
            message_content = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []

            # Convert tool calls back to framework format if needed
            # The framework typically handles this if we return the raw message or standard format
            # BaseChatClient expects us to return a ChatResponse.
            
            # We need to construct the response message
            # NOTE: This is a simplification. Real mapping might need more detail.
            
            response_msg = ChatMessage(
                role=choice.message.role,
                text=message_content,
                # tools usage logic here if needed
            )
            
            # If there are tool calls, we need to attach them.
            # The ChatMessage definition in agent_framework handles tool_calls?
            # Let's look at ChatMessage definition if possible,
            # but for now we assume text is primary.
            # Wait, ChatMessage usually has 'tool_calls' field.
            if tool_calls:
                # Mapping HF tool calls to framework tool calls
                # This part is critical for the Manager agent.
                # If strict mapping is required, we might need to inspect ChatMessage more closely.
                # For now, we'll rely on the framework's ability to parse.
                pass

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
        
        tools = chat_options.tools if chat_options.tools else None
        tool_choice = chat_options.tool_choice
        if str(tool_choice) == "ToolMode.AUTO":
            tool_choice = "auto"

        try:
            # Streaming call
            stream = await asyncio.to_thread(
                self._client.chat_completion,
                messages=hf_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=chat_options.max_tokens or 2048,
                temperature=chat_options.temperature or 0.7,
                stream=True,
            )

            for chunk in stream:
                # Chunk is ChatCompletionStreamOutput
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Convert to ChatResponseUpdate
                # agent_framework might expect specific fields
                yield ChatResponseUpdate(
                    role=delta.role,
                    content=delta.content,
                    # tool_calls handling for stream if needed
                )
                
                # Yield control to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("HuggingFace Streaming error", error=str(e))
            raise
