---
title: DeepBoner
emoji: 🍆
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "6.0.1"
python_version: "3.11"
app_file: src/app.py
pinned: true
license: apache-2.0
short_description: "Deep Research Agent for the Strongest Boners 💪🔬"
tags:
  - mcp-in-action-track-enterprise
  - mcp-hackathon
  - agents
  - sexual-health
  - reproductive-medicine
  - hormone-therapy
  - wellness-research
  - pydantic-ai
  - llamaindex
  - modal
  - pubmed
  - clinical-trials
  - evidence-based
  - multi-agent
---

# DeepBoner 🍆

[![CI](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Obstacle-Is-The-Way/DeepBoner/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/The-Obstacle-Is-The-Way/DeepBoner/branch/main/graph/badge.svg)](https://codecov.io/gh/The-Obstacle-Is-The-Way/DeepBoner)

> **"Peer-reviewed boners only. We take evidence-based arousal very seriously."** 🔬

### AI-Powered Deep Research Agent for Sexual Health

Making the world harder, one PubMed query at a time. 💪

Deep research for sexual wellness, ED treatments, hormone therapy, libido, and reproductive health - **for all genders**. Because everyone deserves rock-solid scientific evidence for their intimate health questions.

## Why DeepBoner?

Sexual health is health. Period. Yet it remains one of the most under-researched and stigmatized areas of medicine. We built DeepBoner to:

- **Break the stigma** - Ask your embarrassing questions to an AI, not a judgmental search engine
- **Get real science** - Every answer backed by peer-reviewed research from PubMed & clinical trials
- **Cover everyone** - ED, libido, hormones, menopause, HSDD, reproductive health - all genders welcome
- **Stay current** - Access the latest clinical trials and preprints, not decade-old WebMD articles

## Features

- 🔍 **Multi-Source Deep Search**: PubMed, ClinicalTrials.gov, Europe PMC - simultaneously
- 🤖 **MCP Integration**: Use our tools from Claude Desktop or any MCP client
- 🔒 **Modal Sandbox**: Secure execution of AI-generated statistical analysis
- 🧠 **Smart Evidence Synthesis**: LLM-powered judge evaluates and synthesizes findings
- ⚡ **Two Modes**: Simple (fast) or Advanced (multi-agent deep dive)
- 🆓 **Free Tier Available**: Works without API keys (HuggingFace Inference)

## Example Queries

Ask anything about sexual health. We don't judge. The science does.

- 💊 "What drugs improve female libido post-menopause?"
- 🧪 "Clinical trials for ED alternatives to PDE5 inhibitors?"
- 🔬 "Evidence for testosterone therapy in women with HSDD?"
- ⚠️ "Drug interactions with sildenafil?"
- 📊 "Latest research on flibanserin efficacy?"
- 🩺 "Non-hormonal treatments for vaginal dryness?"
- 💪 "Natural supplements for erectile function - what actually works?"

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

Built with love (and rigorous methodology) by **The-Obstacle-Is-The-Way**

## Hackathon

**MCP 1st Birthday Hackathon** - Track 2: MCP in Action (Enterprise)

We believe sexual health research deserves the same AI-powered tooling as every other medical domain. This is our contribution to normalizing conversations about intimate health through technology.

## Disclaimer

⚠️ **This is a research tool, not medical advice.** Always consult a healthcare provider for personal medical decisions. We search the literature - we don't replace your doctor.

## Links

- 🔗 [GitHub Repository](https://github.com/The-Obstacle-Is-The-Way/DeepBoner)
- 🐦 [Launch Tweet](https://x.com/VibeCoderMcSwag/status/1994591979070423086)
- 🤗 [HuggingFace Space](https://huggingface.co/spaces/MCP-1st-Birthday/DeepBoner)

---

*"The science is rock solid."* 🪨
