---
title: DeepCritical
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
python_version: "3.11"
app_file: src/app.py
pinned: false
license: mit
tags:
  - mcp-in-action-track-enterprise
  - mcp-hackathon
  - drug-repurposing
  - biomedical-ai
  - pydantic-ai
  - llamaindex
  - modal
---

# DeepCritical

AI-Powered Drug Repurposing Research Agent

## Features

- **Multi-Source Search**: PubMed, ClinicalTrials.gov, bioRxiv/medRxiv
- **MCP Integration**: Use our tools from Claude Desktop or any MCP client
- **Modal Sandbox**: Secure execution of AI-generated statistical code
- **LlamaIndex RAG**: Semantic search and evidence synthesis

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
uv run python src/app.py
```

Open your browser to `http://localhost:7860`.

### 3. Connect via MCP

This application exposes a Model Context Protocol (MCP) server, allowing you to use its search tools directly from Claude Desktop or other MCP clients.

**MCP Server URL**: `http://localhost:7860/gradio_api/mcp/`

**Claude Desktop Configuration**:
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "deepcritical": {
      "url": "http://localhost:7860/gradio_api/mcp/"
    }
  }
}
```

**Available Tools**:
- `search_pubmed`: Search peer-reviewed biomedical literature.
- `search_clinical_trials`: Search ClinicalTrials.gov.
- `search_biorxiv`: Search bioRxiv/medRxiv preprints.
- `search_all`: Search all sources simultaneously.
- `analyze_hypothesis`: Secure statistical analysis using Modal sandboxes.

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

1.  **Search Slice**: Retrieving evidence from PubMed, ClinicalTrials.gov, and bioRxiv.
2.  **Judge Slice**: Evaluating evidence quality using LLMs.
3.  **Orchestrator Slice**: Managing the research loop and UI.

Built with:
- **PydanticAI**: For robust agent interactions.
- **Gradio**: For the streaming user interface.
- **PubMed, ClinicalTrials.gov, bioRxiv**: For biomedical data.
- **MCP**: For universal tool access.
- **Modal**: For secure code execution.

## Team

- The-Obstacle-Is-The-Way
- MarioAderman
- Josephrp

## Links

- [GitHub Repository](https://github.com/The-Obstacle-Is-The-Way/DeepCritical-1)
