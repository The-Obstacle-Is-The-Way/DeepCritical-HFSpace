# Phase 4 Implementation Spec: Orchestrator & UI

**Goal**: Connect the Brain and the Body, then give it a Face.
**Philosophy**: "Streaming is Trust."

---

## 1. The Slice Definition

This slice connects:
1.  **Orchestrator**: The state machine (While loop) calling Search -> Judge.
2.  **UI**: Gradio interface that visualizes the loop.

**Directory**: `src/features/orchestrator/` and `src/app.py`

---

## 2. The Orchestrator Logic

This is the "Agent" logic.

```python
class Orchestrator:
    def __init__(self, search_handler, judge_handler):
        self.search = search_handler
        self.judge = judge_handler
        self.history = []

    async def run_generator(self, query: str):
        """Yields events for the UI"""
        yield AgentEvent("Searching...")
        evidence = await self.search.execute(query)
        
        yield AgentEvent("Judging...")
        assessment = await self.judge.assess(query, evidence)
        
        if assessment.sufficient:
            yield AgentEvent("Complete", data=assessment)
        else:
            yield AgentEvent("Looping...", data=assessment.next_queries)
```

---

## 3. The UI (Gradio)

We use **Gradio 5** generator pattern for real-time feedback.

```python
import gradio as gr

async def interact(message, history):
    agent = Orchestrator(...)
    async for event in agent.run_generator(message):
        yield f"**{event.step}**: {event.details}"

demo = gr.ChatInterface(fn=interact, type="messages")
```

---

## 4. TDD Workflow

### Step 1: Test the State Machine
Test the loop logic without UI.

```python
@pytest.mark.asyncio
async def test_orchestrator_loop_limit():
    # Configure judge to always return "sufficient=False"
    # Assert loop stops at MAX_ITERATIONS
```

### Step 2: Build UI
Run `uv run python src/app.py` and verify locally.

---

## 5. Implementation Checklist

- [ ] Implement `Orchestrator` class.
- [ ] Write loop logic with max_iterations safety.
- [ ] Create `src/app.py` with Gradio.
- [ ] Add "Deployment" configuration (Dockerfile/Spaces config).
