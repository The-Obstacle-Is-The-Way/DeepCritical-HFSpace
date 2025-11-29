# Phase 12 Implementation Spec: MCP Server Integration

**Goal**: Expose DeepBoner search tools as MCP servers for Track 2 compliance.
**Philosophy**: "MCP is the bridge between tools and LLMs."
**Prerequisite**: Phase 11 complete (all search tools working)
**Priority**: P0 - REQUIRED FOR HACKATHON TRACK 2
**Estimated Time**: 2-3 hours

---

## 1. Why MCP Server?

### Hackathon Requirement

| Requirement | Status Before | Status After |
|-------------|---------------|--------------|
| Must use MCP servers as tools | **MISSING** | **COMPLIANT** |
| Autonomous Agent behavior | **Have it** | Have it |
| Must be Gradio app | **Have it** | Have it |
| Planning/reasoning/execution | **Have it** | Have it |

**Bottom Line**: Without MCP server, we're disqualified from Track 2.

### What MCP Enables

```text
Current State:
  Our Tools → Called directly by Python code → Only our app can use them

After MCP:
  Our Tools → Exposed via MCP protocol → Claude Desktop, Cursor, ANY MCP client
```

---

## 2. Implementation Options Analysis

### Option A: Gradio MCP (Recommended)

**Pros:**
- Single parameter: `demo.launch(mcp_server=True)`
- Already have Gradio app
- Automatic tool schema generation from docstrings
- Built into Gradio 5.0+

**Cons:**
- Requires Gradio 5.0+ with MCP extras
- Must follow strict docstring format

### Option B: Native MCP SDK (FastMCP)

**Pros:**
- More control over tool definitions
- Explicit server configuration
- Separate from UI concerns

**Cons:**
- Separate server process
- More code to maintain
- Additional dependency

### Decision: **Gradio MCP (Option A)**

Rationale:
1. Already have Gradio app (`src/app.py`)
2. Minimal code changes
3. Judges will appreciate simplicity
4. Follows hackathon's official Gradio guide

---

## 3. Technical Specification

### 3.1 Dependencies

```toml
# pyproject.toml - add MCP extras
dependencies = [
    "gradio[mcp]>=5.0.0",  # Updated from gradio>=4.0
    # ... existing deps
]
```

### 3.2 MCP Tool Functions

Each tool needs:
1. **Type hints** on all parameters
2. **Docstring** with Args section (Google style)
3. **Return type** annotation
4. **`api_name`** parameter for explicit endpoint naming

```python
async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature.

    Args:
        query: Search query for PubMed (e.g., "metformin alzheimer")
        max_results: Maximum number of results to return (1-50)

    Returns:
        Formatted search results with titles, citations, and abstracts
    """
```

### 3.3 MCP Server URL

Once launched:
```text
http://localhost:7860/gradio_api/mcp/
```

Or on HuggingFace Spaces:
```text
https://[space-id].hf.space/gradio_api/mcp/
```

---

## 4. Implementation

### 4.1 MCP Tool Wrappers (`src/mcp_tools.py`)

```python
"""MCP tool wrappers for DeepBoner search tools.

These functions expose our search tools via MCP protocol.
Each function follows the MCP tool contract:
- Full type hints
- Google-style docstrings with Args section
- Formatted string returns
"""

from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool


# Singleton instances (avoid recreating on each call)
_pubmed = PubMedTool()
_trials = ClinicalTrialsTool()
_europepmc = EuropePMCTool()


async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for peer-reviewed biomedical literature.

    Searches NCBI PubMed database for scientific papers matching your query.
    Returns titles, authors, abstracts, and citation information.

    Args:
        query: Search query (e.g., "metformin alzheimer", "drug repurposing cancer")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted search results with paper titles, authors, dates, and abstracts
    """
    max_results = max(1, min(50, max_results))  # Clamp to valid range

    results = await _pubmed.search(query, max_results)

    if not results:
        return f"No PubMed results found for: {query}"

    formatted = [f"## PubMed Results for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**Authors**: {', '.join(evidence.citation.authors[:3])}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_clinical_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for clinical trial data.

    Searches the ClinicalTrials.gov database for trials matching your query.
    Returns trial titles, phases, status, conditions, and interventions.

    Args:
        query: Search query (e.g., "metformin alzheimer", "diabetes phase 3")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted clinical trial information with NCT IDs, phases, and status
    """
    max_results = max(1, min(50, max_results))

    results = await _trials.search(query, max_results)

    if not results:
        return f"No clinical trials found for: {query}"

    formatted = [f"## Clinical Trials for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_europepmc(query: str, max_results: int = 10) -> str:
    """Search Europe PMC for preprint and open access research.

    Searches Europe PMC for preprints and open access papers.
    Note: Preprints are NOT peer-reviewed but contain the latest findings.

    Args:
        query: Search query (e.g., "metformin neuroprotection", "long covid treatment")
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        Formatted preprint results with titles, authors, and abstracts
    """
    max_results = max(1, min(50, max_results))

    results = await _europepmc.search(query, max_results)

    if not results:
        return f"No Europe PMC results found for: {query}"

    formatted = [f"## Preprint Results for: {query}\n"]
    for i, evidence in enumerate(results, 1):
        formatted.append(f"### {i}. {evidence.citation.title}")
        formatted.append(f"**Authors**: {', '.join(evidence.citation.authors[:3])}")
        formatted.append(f"**Date**: {evidence.citation.date}")
        formatted.append(f"**URL**: {evidence.citation.url}")
        formatted.append(f"\n{evidence.content}\n")

    return "\n".join(formatted)


async def search_all_sources(query: str, max_per_source: int = 5) -> str:
    """Search all biomedical sources simultaneously.

    Performs parallel search across PubMed, ClinicalTrials.gov, and Europe PMC.
    This is the most comprehensive search option for drug repurposing research.

    Args:
        query: Search query (e.g., "metformin alzheimer", "aspirin cancer prevention")
        max_per_source: Maximum results per source (1-20, default 5)

    Returns:
        Combined results from all sources with source labels
    """
    import asyncio

    max_per_source = max(1, min(20, max_per_source))

    # Run all searches in parallel
    pubmed_task = search_pubmed(query, max_per_source)
    trials_task = search_clinical_trials(query, max_per_source)
    europepmc_task = search_europepmc(query, max_per_source)

    pubmed_results, trials_results, europepmc_results = await asyncio.gather(
        pubmed_task, trials_task, europepmc_task, return_exceptions=True
    )

    formatted = [f"# Comprehensive Search: {query}\n"]

    # Add each result section (handle exceptions gracefully)
    if isinstance(pubmed_results, str):
        formatted.append(pubmed_results)
    else:
        formatted.append(f"## PubMed\n*Error: {pubmed_results}*\n")

    if isinstance(trials_results, str):
        formatted.append(trials_results)
    else:
        formatted.append(f"## Clinical Trials\n*Error: {trials_results}*\n")

    if isinstance(europepmc_results, str):
        formatted.append(europepmc_results)
    else:
        formatted.append(f"## Preprints\n*Error: {europepmc_results}*\n")

    return "\n---\n".join(formatted)
```

### 4.2 Update Gradio App (`src/app.py`)

```python
"""Gradio UI for DeepBoner agent with MCP server support."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr

from src.agent_factory.judges import JudgeHandler, MockJudgeHandler
from src.mcp_tools import (
    search_all_sources,
    search_europepmc,
    search_clinical_trials,
    search_pubmed,
)
from src.orchestrator_factory import create_orchestrator
from src.tools.europepmc import EuropePMCTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.models import OrchestratorConfig


# ... (existing configure_orchestrator and research_agent functions unchanged)


def create_demo() -> Any:
    """
    Create the Gradio demo interface with MCP support.

    Returns:
        Configured Gradio Blocks interface with MCP server enabled
    """
    with gr.Blocks(
        title="DeepBoner - Drug Repurposing Research Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # DeepBoner
        ## AI-Powered Drug Repurposing Research Agent

        Ask questions about potential drug repurposing opportunities.
        The agent searches PubMed, ClinicalTrials.gov, and Europe PMC preprints.

        **Example questions:**
        - "What drugs could be repurposed for Alzheimer's disease?"
        - "Is metformin effective for cancer treatment?"
        - "What existing medications show promise for Long COVID?"
        """)

        # Main chat interface (existing)
        gr.ChatInterface(
            fn=research_agent,
            type="messages",
            title="",
            examples=[
                "What drugs could be repurposed for Alzheimer's disease?",
                "Is metformin effective for treating cancer?",
                "What medications show promise for Long COVID treatment?",
                "Can statins be repurposed for neurological conditions?",
            ],
            additional_inputs=[
                gr.Radio(
                    choices=["simple", "magentic"],
                    value="simple",
                    label="Orchestrator Mode",
                    info="Simple: Linear (OpenAI/Anthropic) | Magentic: Multi-Agent (OpenAI)",
                )
            ],
        )

        # MCP Tool Interfaces (exposed via MCP protocol)
        gr.Markdown("---\n## MCP Tools (Also Available via Claude Desktop)")

        with gr.Tab("PubMed Search"):
            gr.Interface(
                fn=search_pubmed,
                inputs=[
                    gr.Textbox(label="Query", placeholder="metformin alzheimer"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_pubmed",
            )

        with gr.Tab("Clinical Trials"):
            gr.Interface(
                fn=search_clinical_trials,
                inputs=[
                    gr.Textbox(label="Query", placeholder="diabetes phase 3"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_clinical_trials",
            )

        with gr.Tab("Preprints"):
            gr.Interface(
                fn=search_europepmc,
                inputs=[
                    gr.Textbox(label="Query", placeholder="long covid treatment"),
                    gr.Slider(1, 50, value=10, step=1, label="Max Results"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_europepmc",
            )

        with gr.Tab("Search All"):
            gr.Interface(
                fn=search_all_sources,
                inputs=[
                    gr.Textbox(label="Query", placeholder="metformin cancer"),
                    gr.Slider(1, 20, value=5, step=1, label="Max Per Source"),
                ],
                outputs=gr.Markdown(label="Results"),
                api_name="search_all",
            )

        gr.Markdown("""
        ---
        **Note**: This is a research tool and should not be used for medical decisions.
        Always consult healthcare professionals for medical advice.

        Built with PydanticAI + PubMed, ClinicalTrials.gov & Europe PMC

        **MCP Server**: Available at `/gradio_api/mcp/` for Claude Desktop integration
        """)

    return demo


def main() -> None:
    """Run the Gradio app with MCP server enabled."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True,  # Enable MCP server
    )


if __name__ == "__main__":
    main()
```

---

## 5. TDD Test Suite

### 5.1 Unit Tests (`tests/unit/test_mcp_tools.py`)

```python
"""Unit tests for MCP tool wrappers."""

from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_tools import (
    search_all_sources,
    search_europepmc,
    search_clinical_trials,
    search_pubmed,
)
from src.utils.models import Citation, Evidence


@pytest.fixture
def mock_evidence() -> Evidence:
    """Sample evidence for testing."""
    return Evidence(
        content="Metformin shows neuroprotective effects in preclinical models.",
        citation=Citation(
            source="pubmed",
            title="Metformin and Alzheimer's Disease",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            date="2024-01-15",
            authors=["Smith J", "Jones M", "Brown K"],
        ),
        relevance=0.85,
    )


class TestSearchPubMed:
    """Tests for search_pubmed MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, mock_evidence: Evidence) -> None:
        """Should return formatted markdown string."""
        with patch("src.mcp_tools._pubmed") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[mock_evidence])

            result = await search_pubmed("metformin alzheimer", 10)

            assert isinstance(result, str)
            assert "PubMed Results" in result
            assert "Metformin and Alzheimer's Disease" in result
            assert "Smith J" in result

    @pytest.mark.asyncio
    async def test_clamps_max_results(self) -> None:
        """Should clamp max_results to valid range (1-50)."""
        with patch("src.mcp_tools._pubmed") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[])

            # Test lower bound
            await search_pubmed("test", 0)
            mock_tool.search.assert_called_with("test", 1)

            # Test upper bound
            await search_pubmed("test", 100)
            mock_tool.search.assert_called_with("test", 50)

    @pytest.mark.asyncio
    async def test_handles_no_results(self) -> None:
        """Should return appropriate message when no results."""
        with patch("src.mcp_tools._pubmed") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[])

            result = await search_pubmed("xyznonexistent", 10)

            assert "No PubMed results found" in result


class TestSearchClinicalTrials:
    """Tests for search_clinical_trials MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, mock_evidence: Evidence) -> None:
        """Should return formatted markdown string."""
        mock_evidence.citation.source = "clinicaltrials"  # type: ignore

        with patch("src.mcp_tools._trials") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[mock_evidence])

            result = await search_clinical_trials("diabetes", 10)

            assert isinstance(result, str)
            assert "Clinical Trials" in result


class TestSearchEuropePMC:
    """Tests for search_europepmc MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, mock_evidence: Evidence) -> None:
        """Should return formatted markdown string."""
        mock_evidence.citation.source = "europepmc"  # type: ignore

        with patch("src.mcp_tools._europepmc") as mock_tool:
            mock_tool.search = AsyncMock(return_value=[mock_evidence])

            result = await search_europepmc("preprint search", 10)

            assert isinstance(result, str)
            assert "Preprint Results" in result


class TestSearchAllSources:
    """Tests for search_all_sources MCP tool."""

    @pytest.mark.asyncio
    async def test_combines_all_sources(self, mock_evidence: Evidence) -> None:
        """Should combine results from all sources."""
        with patch("src.mcp_tools.search_pubmed", new_callable=AsyncMock) as mock_pubmed, \
             patch("src.mcp_tools.search_clinical_trials", new_callable=AsyncMock) as mock_trials, \
             patch("src.mcp_tools.search_europepmc", new_callable=AsyncMock) as mock_europepmc:

            mock_pubmed.return_value = "## PubMed Results"
            mock_trials.return_value = "## Clinical Trials"
            mock_europepmc.return_value = "## Preprints"

            result = await search_all_sources("metformin", 5)

            assert "Comprehensive Search" in result
            assert "PubMed" in result
            assert "Clinical Trials" in result
            assert "Preprints" in result

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self) -> None:
        """Should handle partial failures gracefully."""
        with patch("src.mcp_tools.search_pubmed", new_callable=AsyncMock) as mock_pubmed, \
             patch("src.mcp_tools.search_clinical_trials", new_callable=AsyncMock) as mock_trials, \
             patch("src.mcp_tools.search_europepmc", new_callable=AsyncMock) as mock_europepmc:

            mock_pubmed.return_value = "## PubMed Results"
            mock_trials.side_effect = Exception("API Error")
            mock_europepmc.return_value = "## Preprints"

            result = await search_all_sources("metformin", 5)

            # Should still contain working sources
            assert "PubMed" in result
            assert "Preprints" in result
            # Should show error for failed source
            assert "Error" in result


class TestMCPDocstrings:
    """Tests that docstrings follow MCP format."""

    def test_search_pubmed_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_pubmed.__doc__ is not None
        assert "Args:" in search_pubmed.__doc__
        assert "query:" in search_pubmed.__doc__
        assert "max_results:" in search_pubmed.__doc__
        assert "Returns:" in search_pubmed.__doc__

    def test_search_clinical_trials_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_clinical_trials.__doc__ is not None
        assert "Args:" in search_clinical_trials.__doc__

    def test_search_europepmc_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_europepmc.__doc__ is not None
        assert "Args:" in search_europepmc.__doc__

    def test_search_all_sources_has_args_section(self) -> None:
        """Docstring must have Args section for MCP schema generation."""
        assert search_all_sources.__doc__ is not None
        assert "Args:" in search_all_sources.__doc__


class TestMCPTypeHints:
    """Tests that type hints are complete for MCP."""

    def test_search_pubmed_type_hints(self) -> None:
        """All parameters and return must have type hints."""
        import inspect

        sig = inspect.signature(search_pubmed)

        # Check parameter hints
        assert sig.parameters["query"].annotation == str
        assert sig.parameters["max_results"].annotation == int

        # Check return hint
        assert sig.return_annotation == str

    def test_search_clinical_trials_type_hints(self) -> None:
        """All parameters and return must have type hints."""
        import inspect

        sig = inspect.signature(search_clinical_trials)
        assert sig.parameters["query"].annotation == str
        assert sig.parameters["max_results"].annotation == int
        assert sig.return_annotation == str
```

### 5.2 Integration Test (`tests/integration/test_mcp_server.py`)

```python
"""Integration tests for MCP server functionality."""

import pytest


class TestMCPServerIntegration:
    """Integration tests for MCP server (requires running app)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcp_tools_work_end_to_end(self) -> None:
        """Test that MCP tools execute real searches."""
        from src.mcp_tools import search_pubmed

        result = await search_pubmed("metformin diabetes", 3)

        assert isinstance(result, str)
        assert "PubMed Results" in result
        # Should have actual content (not just "no results")
        assert len(result) > 100
```

---

## 6. Claude Desktop Configuration

### 6.1 Local Development

```json
// ~/.config/claude/claude_desktop_config.json (Linux/Mac)
// %APPDATA%\Claude\claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "deepboner": {
      "url": "http://localhost:7860/gradio_api/mcp/"
    }
  }
}
```

### 6.2 HuggingFace Spaces

```json
{
  "mcpServers": {
    "deepboner": {
      "url": "https://your-space.hf.space/gradio_api/mcp/"
    }
  }
}
```

### 6.3 Private Spaces (with auth)

```json
{
  "mcpServers": {
    "deepboner": {
      "url": "https://your-space.hf.space/gradio_api/mcp/",
      "headers": {
        "Authorization": "Bearer hf_xxxxxxxxxxxxx"
      }
    }
  }
}
```

---

## 7. Verification Commands

```bash
# 1. Install MCP extras
uv add "gradio[mcp]>=5.0.0"

# 2. Run unit tests
uv run pytest tests/unit/test_mcp_tools.py -v

# 3. Run full test suite
make check

# 4. Start server with MCP
uv run python src/app.py

# 5. Verify MCP schema (in another terminal)
curl http://localhost:7860/gradio_api/mcp/schema

# 6. Test with MCP Inspector
npx @anthropic/mcp-inspector http://localhost:7860/gradio_api/mcp/

# 7. Integration test (requires running server)
uv run pytest tests/integration/test_mcp_server.py -v -m integration
```

---

## 8. Definition of Done

Phase 12 is **COMPLETE** when:

- [ ] `src/mcp_tools.py` created with all 4 MCP tools
- [ ] `src/app.py` updated with `mcp_server=True`
- [ ] Unit tests in `tests/unit/test_mcp_tools.py`
- [ ] Integration test in `tests/integration/test_mcp_server.py`
- [ ] `pyproject.toml` updated with `gradio[mcp]`
- [ ] MCP schema accessible at `/gradio_api/mcp/schema`
- [ ] Claude Desktop can connect and use tools
- [ ] All unit tests pass
- [ ] Lints pass

---

## 9. Demo Script for Judges

### Show MCP Integration Works

1. **Start the server**:
   ```bash
   uv run python src/app.py
   ```

2. **Show Claude Desktop using our tools**:
   - Open Claude Desktop with DeepBoner MCP configured
   - Ask: "Search PubMed for metformin Alzheimer's"
   - Show real results appearing
   - Ask: "Now search clinical trials for the same"
   - Show combined analysis

3. **Show MCP Inspector**:
   ```bash
   npx @anthropic/mcp-inspector http://localhost:7860/gradio_api/mcp/
   ```
   - Show all 4 tools listed
   - Execute `search_pubmed` from inspector
   - Show results

---

## 10. Value Delivered

| Before | After |
|--------|-------|
| Tools only usable in our app | Tools usable by ANY MCP client |
| Not Track 2 compliant | **FULLY TRACK 2 COMPLIANT** |
| Can't use with Claude Desktop | Full Claude Desktop integration |

**Prize Impact**:
- Without MCP: **Disqualified from Track 2**
- With MCP: **Eligible for $2,500 1st place**

---

## 11. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/mcp_tools.py` | CREATE | MCP tool wrapper functions |
| `src/app.py` | MODIFY | Add `mcp_server=True`, add tool tabs |
| `pyproject.toml` | MODIFY | Add `gradio[mcp]>=5.0.0` |
| `tests/unit/test_mcp_tools.py` | CREATE | Unit tests for MCP tools |
| `tests/integration/test_mcp_server.py` | CREATE | Integration tests |
| `README.md` | MODIFY | Add MCP usage instructions |

---

## 12. Architecture After Phase 12

```text
┌────────────────────────────────────────────────────────────────┐
│                      Claude Desktop / Cursor                   │
│                           (MCP Client)                         │
└─────────────────────────────┬──────────────────────────────────┘
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio MCP Server                        │
│                  /gradio_api/mcp/                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │search_pubmed │ │search_trials │ │search_epmc   │ │search_  │ │
│  │              │ │              │ │              │ │all      │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └────┬────┘ │
└─────────┼────────────────┼────────────────┼──────────────┼──────┘
          │                │                │              │
          ▼                ▼                ▼              ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐    (calls all)
   │PubMedTool│     │Trials    │     │EuropePMC │
   │          │     │Tool      │     │Tool      │
   └──────────┘     └──────────┘     └──────────┘
```

**This is the MCP compliance stack.**
