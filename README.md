---
title: DeepCritical
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: src/app.py
pinned: false
license: mit
---

# DeepCritical

AI-Powered Drug Repurposing Research Agent

## Quick Start

### 1. Environment Setup

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync
```

### 2. Run the UI

```bash
# Start the Gradio app
uv run python -m src.app
```

Open your browser to `http://localhost:7860`.

## Development

### Run Tests

```bash
uv run pytest
```

### Run Checks

```bash
make check
```

## Architecture

DeepCritical uses a Vertical Slice Architecture:

1.  **Search Slice**: Retrieving evidence from PubMed and the Web.
2.  **Judge Slice**: Evaluating evidence quality using LLMs.
3.  **Orchestrator Slice**: Managing the research loop and UI.

Built with:
- **PydanticAI**: For robust agent interactions.
- **Gradio**: For the streaming user interface.
- **PubMed**: For biomedical literature.
- **DuckDuckGo**: For general web search.

