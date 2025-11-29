---
title: DeepBoner
emoji: 🍆
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "6.0.1"
python_version: "3.11"
app_file: src/app.py
pinned: false
license: mit
tags:
  - sexual-health
  - reproductive-medicine
  - hormone-therapy
  - wellness-research
  - mcp-hackathon
  - pydantic-ai
  - llamaindex
  - modal
---

# DeepBoner 🍆

AI-Native Sexual Health Research Agent

Deep research for sexual wellness, ED treatments, hormone therapy, libido, and reproductive health - for all genders.

## Features

- **Multi-Source Search**: PubMed, ClinicalTrials.gov, Europe PMC
- **MCP Integration**: Use our tools from Claude Desktop or any MCP client
- **Modal Sandbox**: Secure execution of AI-generated statistical code
- **LlamaIndex RAG**: Semantic search and evidence synthesis

## Example Queries

- "What drugs improve female libido post-menopause?"
- "Clinical trials for erectile dysfunction alternatives to PDE5 inhibitors?"
- "Evidence for testosterone therapy in women with HSDD?"
- "Drug interactions with sildenafil?"
- "What's the latest research on flibanserin efficacy?"

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
    "deepboner": {
      "url": "http://localhost:7860/gradio_api/mcp/"
    }
  }
}
```

**Available Tools**:
- `search_pubmed`: Search peer-reviewed biomedical literature.
- `search_clinical_trials`: Search ClinicalTrials.gov.
- `search_europepmc`: Search Europe PMC preprints and papers.
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

DeepBoner uses a Vertical Slice Architecture:

1.  **Search Slice**: Retrieving evidence from PubMed, ClinicalTrials.gov, and Europe PMC.
2.  **Judge Slice**: Evaluating evidence quality using LLMs.
3.  **Orchestrator Slice**: Managing the research loop and UI.

Built with:
- **PydanticAI**: For robust agent interactions.
- **Gradio**: For the streaming user interface.
- **PubMed, ClinicalTrials.gov, Europe PMC**: For biomedical data.
- **MCP**: For universal tool access.
- **Modal**: For secure code execution.

## Team

- The-Obstacle-Is-The-Way
- MarioAderman
- EmployeeNo427

## Links

- [GitHub Repository](https://github.com/The-Obstacle-Is-The-Way/DeepBoner)
