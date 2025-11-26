# MCP Server Integration

## Priority: P0 - REQUIRED FOR TRACK 2

> **✅ STATUS: IMPLEMENTED** - See `src/mcp_tools.py` and `src/app.py`
> MCP endpoint: `/gradio_api/mcp/`

---

## What We Need

Expose our search tools as MCP servers so Claude Desktop/Cursor can use them.

### Current Tools to Expose

| Tool | File | MCP Tool Name |
|------|------|---------------|
| PubMed Search | `src/tools/pubmed.py` | `search_pubmed` |
| ClinicalTrials Search | `src/tools/clinicaltrials.py` | `search_clinical_trials` |
| bioRxiv Search | `src/tools/biorxiv.py` | `search_biorxiv` |
| Combined Search | `src/tools/search_handler.py` | `search_all_sources` |

---

## Implementation Options

### Option 1: Gradio MCP (Recommended)

Gradio 5.0+ can expose any Gradio app as an MCP server automatically.

```python
# src/mcp_server.py
import gradio as gr
from src.tools.pubmed import PubMedTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.biorxiv import BioRxivTool

pubmed = PubMedTool()
trials = ClinicalTrialsTool()
biorxiv = BioRxivTool()

async def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature."""
    results = await pubmed.search(query, max_results)
    return "\n\n".join([f"**{e.citation.title}**\n{e.content}" for e in results])

async def search_clinical_trials(query: str, max_results: int = 10) -> str:
    """Search ClinicalTrials.gov for clinical trial data."""
    results = await trials.search(query, max_results)
    return "\n\n".join([f"**{e.citation.title}**\n{e.content}" for e in results])

async def search_biorxiv(query: str, max_results: int = 10) -> str:
    """Search bioRxiv/medRxiv for preprints."""
    results = await biorxiv.search(query, max_results)
    return "\n\n".join([f"**{e.citation.title}**\n{e.content}" for e in results])

# Create Gradio interface
demo = gr.Interface(
    fn=[search_pubmed, search_clinical_trials, search_biorxiv],
    inputs=[gr.Textbox(label="Query"), gr.Number(label="Max Results", value=10)],
    outputs=gr.Textbox(label="Results"),
)

# Launch as MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True)  # Gradio 5.0+ feature
```

### Option 2: Native MCP SDK

Use the official MCP Python SDK:

```bash
uv add mcp
```

```python
# src/mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

from src.tools.pubmed import PubMedTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.biorxiv import BioRxivTool

server = Server("deepcritical-research")

@server.tool()
async def search_pubmed(query: str, max_results: int = 10) -> list[TextContent]:
    """Search PubMed for biomedical literature on drug repurposing."""
    tool = PubMedTool()
    results = await tool.search(query, max_results)
    return [TextContent(type="text", text=e.content) for e in results]

@server.tool()
async def search_clinical_trials(query: str, max_results: int = 10) -> list[TextContent]:
    """Search ClinicalTrials.gov for clinical trials."""
    tool = ClinicalTrialsTool()
    results = await tool.search(query, max_results)
    return [TextContent(type="text", text=e.content) for e in results]

@server.tool()
async def search_biorxiv(query: str, max_results: int = 10) -> list[TextContent]:
    """Search bioRxiv/medRxiv for preprints (not peer-reviewed)."""
    tool = BioRxivTool()
    results = await tool.search(query, max_results)
    return [TextContent(type="text", text=e.content) for e in results]

if __name__ == "__main__":
    server.run()
```

---

## Claude Desktop Configuration

After implementing, users add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "deepcritical": {
      "command": "uv",
      "args": ["run", "python", "src/mcp_server.py"],
      "cwd": "/path/to/DeepCritical-1"
    }
  }
}
```

---

## Testing MCP Server

1. Start the MCP server (via Gradio app):

```bash
uv run python src/app.py
```

2. Check MCP schema:

```bash
curl http://localhost:7860/gradio_api/mcp/schema | jq
```

3. Test with MCP Inspector:

```bash
npx @anthropic/mcp-inspector http://localhost:7860/gradio_api/mcp/sse
```

4. Verify tools appear and work

---

## Demo Video Script

For the hackathon submission video:

1. Show Claude Desktop with DeepCritical MCP tools
2. Ask: "Search PubMed for metformin Alzheimer's"
3. Show real results appearing
4. Ask: "Now search clinical trials for the same"
5. Show combined analysis

This proves MCP integration works.

---

## Files Created

- [x] `src/mcp_tools.py` - MCP tool wrapper functions
- [x] `src/app.py` - Gradio app with `mcp_server=True`
- [x] `tests/unit/test_mcp_tools.py` - Unit tests
- [x] `tests/integration/test_mcp_tools_live.py` - Integration tests
- [x] `README.md` - Updated with MCP usage instructions
