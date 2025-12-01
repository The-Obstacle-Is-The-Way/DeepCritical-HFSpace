# P0 Bug: AIFunction Not JSON Serializable (Free Tier Broken)

**Severity**: P0 (Critical) - Free Tier cannot perform research
**Status**: RESOLVED
**Discovered**: 2025-12-01
**Resolved**: 2025-12-01
**Reporter**: Production user via HuggingFace Spaces

## Symptom

Every search round fails with:
```
📚 SEARCH_COMPLETE: searcher: Agent searcher: Error processing request -
Object of type AIFunction is not JSON serializable
```

Research never completes. Users see 5 rounds of the same error.

## Root Cause

### The Problem

In `src/clients/huggingface.py` lines 82-103:

```python
# Extract tool configuration
tools = chat_options.tools if chat_options.tools else None  # AIFunction objects!
...
call_fn = partial(
    self._client.chat_completion,
    messages=hf_messages,
    tools=tools,  # <-- RAW AIFunction objects passed here
    ...
)
```

The `chat_options.tools` contains `AIFunction` objects from Microsoft's agent-framework.
When `requests` tries to serialize these for the HTTP request, it fails:
```
TypeError: Object of type AIFunction is not JSON serializable
```

### Why This Happens

1. Microsoft's agent-framework defines tools as `AIFunction` objects
2. `ChatAgent` with tools passes them via `chat_options.tools`
3. Our `HuggingFaceChatClient` forwards them directly to `InferenceClient.chat_completion()`
4. `requests.post()` internally calls `json.dumps()` on the request body
5. `AIFunction` has no `__json__()` method or isn't a dict → TypeError

## Impact

| Component | Impact |
|-----------|--------|
| Free Tier (HuggingFace) | **COMPLETELY BROKEN** |
| Advanced Mode without API key | **Cannot do research** |
| Paid Tier (OpenAI) | Unaffected (OpenAI handles AIFunction) |

## Professional Fix (Full Implementation)

Qwen2.5-72B-Instruct **SUPPORTS** function calling via HuggingFace. The fix requires:

1. **Request Serialization**: Convert `AIFunction` → OpenAI-compatible JSON
2. **Response Parsing**: Convert HuggingFace `tool_calls` → Framework `FunctionCallContent`

### Part 1: Tool Serialization (`_convert_tools`)

```python
def _convert_tools(self, tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert AIFunction objects to OpenAI-compatible tool definitions.

    AIFunction.to_dict() returns:
        {'type': 'ai_function', 'name': '...', 'description': '...', 'input_model': {...}}

    OpenAI/HuggingFace expects:
        {'type': 'function', 'function': {'name': '...', 'description': '...', 'parameters': {...}}}
    """
    if not tools:
        return None

    json_tools = []
    for tool in tools:
        if hasattr(tool, 'to_dict'):
            t_dict = tool.to_dict()
            json_tools.append({
                "type": "function",
                "function": {
                    "name": t_dict["name"],
                    "description": t_dict.get("description", ""),
                    "parameters": t_dict["input_model"]
                }
            })
        elif isinstance(tool, dict):
            json_tools.append(tool)
        else:
            logger.warning(f"Skipping non-serializable tool: {type(tool)}")

    return json_tools if json_tools else None
```

### Part 2: Response Parsing (Tool Calls → FunctionCallContent)

When HuggingFace returns tool calls, we must convert them to the framework's format:

```python
from agent_framework._types import FunctionCallContent

# In _inner_get_response, after getting the response:
choice = choices[0]
message = choice.message
message_content = message.content or ""

# Parse tool calls if present
contents: list[Any] = []
if hasattr(message, 'tool_calls') and message.tool_calls:
    for tc in message.tool_calls:
        # HF returns: tc.id, tc.function.name, tc.function.arguments
        contents.append(FunctionCallContent(
            call_id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments  # JSON string or dict
        ))

response_msg = ChatMessage(
    role=cast(Any, message.role),
    text=message_content,
    contents=contents if contents else None
)
```

### Verified Schema Mapping

```python
# AIFunction.to_dict() output (verified 2025-12-01):
{
  "type": "ai_function",
  "name": "search_pubmed",
  "description": "Search PubMed for biomedical research papers...",
  "input_model": {
    "properties": {"query": {"title": "Query", "type": "string"}, ...},
    "required": ["query"],
    "type": "object"
  }
}

# Mapped to OpenAI format:
{
  "type": "function",
  "function": {
    "name": "search_pubmed",
    "description": "Search PubMed for biomedical research papers...",
    "parameters": {
      "properties": {"query": {"title": "Query", "type": "string"}, ...},
      "required": ["query"],
      "type": "object"
    }
  }
}
```

## Call Stack Trace

```
User Query (HuggingFace Spaces)
    ↓
src/app.py:research_agent()
    ↓
src/orchestrators/advanced.py:AdvancedOrchestrator.run()
    ↓
agent_framework.MagenticBuilder.run_stream()
    ↓
agent_framework.ChatAgent (SearchAgent with tools=[search_pubmed, ...])
    ↓
src/clients/huggingface.py:HuggingFaceChatClient._inner_get_response()
    → chat_options.tools contains AIFunction objects
    ↓
huggingface_hub.InferenceClient.chat_completion(tools=tools)
    ↓
requests.post(json={..., "tools": [AIFunction, ...]})
    ↓
json.dumps() → TypeError: Object of type AIFunction is not JSON serializable
```

## Testing

```bash
# Reproduce locally (remove OpenAI key)
unset OPENAI_API_KEY
uv run python -c "
import asyncio
from src.orchestrators.advanced import AdvancedOrchestrator

async def test():
    orch = AdvancedOrchestrator(max_rounds=2)
    async for event in orch.run('testosterone benefits'):
        print(f'[{event.type}] {str(event.message)[:50]}...')

asyncio.run(test())
"

# Expected BEFORE fix: TypeError: Object of type AIFunction is not JSON serializable
# Expected AFTER fix: Research completes with tool calls working
```

## Resolution

Implemented full function calling support for HuggingFace client:

1.  **Request Serialization**: Added `_convert_tools` to map `AIFunction` schemas to OpenAI-compatible JSON.
2.  **Response Parsing (Sync)**: Added `_parse_tool_calls` to convert HF `tool_calls` to `FunctionCallContent`.
3.  **Response Parsing (Async)**: Implemented tool call accumulator in `_inner_get_streaming_response` to handle partial tool call deltas and yield valid `FunctionCallContent` objects.

## Verification

Verified with unit tests and manual simulation:

1.  **Serialization**: Confirmed `AIFunction` -> JSON conversion works for `search_pubmed`.
2.  **Streaming**: Verified that fragmented tool call deltas (e.g., `{"query":` then `"testosterone"}`) are correctly reassembled into a single `FunctionCallContent`.
3.  **Integration**: Passed project-level `make check`.

## References

- [HuggingFace Chat Completion - Function Calling](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
- [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Microsoft Agent Framework - AIFunction](https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.aifunction)
