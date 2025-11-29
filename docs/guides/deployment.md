# Deployment Guide
## Launching DeepBoner: Gradio, MCP, & Modal

---

## Overview

DeepBoner is designed for a multi-platform deployment strategy to maximize hackathon impact:

1. **HuggingFace Spaces**: Host the Gradio UI (User Interface).
2. **MCP Server**: Expose research tools to Claude Desktop/Agents.
3. **Modal (Optional)**: Run heavy inference or local LLMs if API costs are prohibitive.

---

## 1. HuggingFace Spaces (Gradio UI)

**Goal**: A public URL where judges/users can try the research agent.

### Prerequisites
- HuggingFace Account
- `gradio` installed (`uv add gradio`)

### Steps

1. **Create Space**:
   - Go to HF Spaces -> Create New Space.
   - SDK: **Gradio**.
   - Hardware: **CPU Basic** (Free) is sufficient (since we use APIs).

2. **Prepare Files**:
   - Ensure `app.py` contains the Gradio interface construction.
   - Ensure `requirements.txt` or `pyproject.toml` lists all dependencies.

3. **Secrets**:
   - Go to Space Settings -> **Repository secrets**.
   - Add `ANTHROPIC_API_KEY` (or your chosen LLM provider key).
   - Add `BRAVE_API_KEY` (for web search).

4. **Deploy**:
   - Push code to the Space's git repo.
   - Watch "Build" logs.

### Streaming Optimization
Ensure `app.py` uses generator functions for the chat interface to prevent timeouts:
```python
# app.py
def predict(message, history):
    agent = ResearchAgent()
    for update in agent.research_stream(message):
        yield update
```

---

## 2. MCP Server Deployment

**Goal**: Allow other agents (like Claude Desktop) to use our PubMed/Research tools directly.

### Local Usage (Claude Desktop)

1. **Install**:
   ```bash
   uv sync
   ```

2. **Configure Claude Desktop**:
   Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "deepboner": {
         "command": "uv",
         "args": ["run", "fastmcp", "run", "src/mcp_servers/pubmed_server.py"],
         "cwd": "/absolute/path/to/DeepBoner"
       }
     }
   }
   ```

3. **Restart Claude**: You should see a 🔌 icon indicating connected tools.

### Remote Deployment (Smithery/Glama)
*Target for "MCP Track" bonus points.*

1. **Dockerize**: Create a `Dockerfile` for the MCP server.
   ```dockerfile
   FROM python:3.11-slim
   COPY . /app
   RUN pip install fastmcp httpx
   CMD ["fastmcp", "run", "src/mcp_servers/pubmed_server.py", "--transport", "sse"]
   ```
   *Note: Use SSE transport for remote/HTTP servers.*

2. **Deploy**: Host on Fly.io or Railway.

---

## 3. Modal (GPU/Heavy Compute)

**Goal**: Run a local LLM (e.g., Llama-3-70B) or handle massive parallel searches if APIs are too slow/expensive.

### Setup
1. **Install**: `uv add modal`
2. **Auth**: `modal token new`

### Logic
Instead of calling Anthropic API, we call a Modal function:

```python
# src/llm/modal_client.py
import modal

stub = modal.Stub("deepboner-inference")

@stub.function(gpu="A100")
def generate_text(prompt: str):
    # Load vLLM or similar
    ...
```

### When to use?
- **Hackathon Demo**: Stick to Anthropic/OpenAI APIs for speed/reliability.
- **Production/Stretch**: Use Modal if you hit rate limits or want to show off "Open Source Models" capability.

---

## Deployment Checklist

### Pre-Flight
- [ ] Run `pytest -m unit` to ensure logic is sound.
- [ ] Run `pytest -m e2e` (one pass) to verify APIs connect.
- [ ] Check `requirements.txt` matches `pyproject.toml`.

### Secrets Management
- [ ] **NEVER** commit `.env` files.
- [ ] Verify keys are added to HF Space settings.

### Post-Launch
- [ ] Test the live URL.
- [ ] Verify "Stop" button in Gradio works (interrupts the agent).
- [ ] Record a walkthrough video (crucial for hackathon submission).
