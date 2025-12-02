# SPEC 17: Accumulator Pattern for Agent Events

**Status**: IMPLEMENTED
**Created**: 2025-12-02
**Author**: AI Agent
**Related**: P0_REPR_BUG_ROOT_CAUSE_ANALYSIS.md

## 1. Context

The Microsoft Agent Framework event model has a specific intended usage pattern:
- `MagenticAgentDeltaEvent.text` → **Content Source** (Streaming)
- `MagenticAgentMessageEvent` → **Completion Signal** (End of Turn)

Our previous implementation incorrectly attempted to extract content from `MagenticAgentMessageEvent.message`. This property is not designed for content extraction and can contain internal representation data (repr strings) for tool-only messages. This led to the "repr bug" where users saw raw Python object strings in the UI.

The **Accumulator Pattern** aligns our codebase with Microsoft's intended architecture (as demonstrated in their `04_magentic_one.py` sample) and resolves the display issues by using the correct event data source.

## 2. The Solution: Accumulator Pattern

Instead of relying on the final message event for content, we adopt the **Accumulator Pattern**, which aligns with the Microsoft Agent Framework's intended usage (as seen in their sample `04_magentic_one.py`).

### 2.1 Core Concept

1.  **Streaming is Truth**: `MagenticAgentDeltaEvent` is the exclusive source of text content. These events are not affected by the upstream bug.
2.  **Accumulation**: The orchestrator maintains a stateful buffer (`current_message_buffer`) that appends text from delta events.
3.  **Signal Processing**: `MagenticAgentMessageEvent` is treated solely as a completion signal ("end of turn"). When received, we consume the buffer to form the final UI message and then clear the buffer.

### 2.2 Logic Flow

```python
current_message_buffer = ""

for event in stream:
    if event is DeltaEvent:
        current_message_buffer += event.text
        emit_streaming_event(event.text)
    
    elif event is MessageEvent:
        # IGNORE event.message (it might be corrupted)
        final_text = current_message_buffer
        if not final_text:
             final_text = "Action completed (Tool Call)"
        
        emit_complete_event(final_text)
        current_message_buffer = ""
```

## 3. Test Plan

To verify this pattern ensures correct output regardless of upstream bugs, we define the following test scenarios:

### 3.1 Scenario A: Standard Text Message
-   **Input**: Sequence of `MagenticAgentDeltaEvent` (with text parts) -> `MagenticAgentMessageEvent` (with corrupted repr).
-   **Expected Output**: The `AgentEvent` emitted at the end must contain the concatenated text from the deltas, NOT the repr string.

### 3.2 Scenario B: Tool Call (No Text)
-   **Input**: No text deltas -> `MagenticAgentMessageEvent` (with corrupted repr).
-   **Expected Output**: The `AgentEvent` should contain a fallback message (e.g., "Action completed (Tool Call)"), NOT the repr string.

## 4. Implementation Details

The pattern is implemented in `src/orchestrators/advanced.py` within the `run()` method loop. It bypasses `_process_event` for these specific event types to ensure strict control over data flow.
