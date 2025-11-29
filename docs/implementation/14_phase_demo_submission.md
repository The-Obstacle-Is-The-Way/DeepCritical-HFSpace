# Phase 14 Implementation Spec: Demo Video & Hackathon Submission

**Goal**: Create compelling demo video and complete hackathon submission.
**Philosophy**: "Ship it with style."
**Prerequisite**: Phases 12-13 complete (MCP + Modal working)
**Priority**: P0 - REQUIRED FOR SUBMISSION
**Deadline**: November 30, 2025 11:59 PM UTC
**Estimated Time**: 2-3 hours

---

## 1. Submission Requirements

### MCP's 1st Birthday Hackathon Checklist

| Requirement | Status | Action |
|-------------|--------|--------|
| HuggingFace Space in `MCP-1st-Birthday` org | Pending | Transfer or create |
| Track tag in README.md | Pending | Add tag |
| Social media post link | Pending | Create post |
| Demo video (1-5 min) | Pending | Record |
| Team members registered | Pending | Verify |
| Original work (Nov 14-30) | **DONE** | All commits in range |

### Track 2: MCP in Action - Tags

```yaml
# Add to HuggingFace Space README.md
tags:
  - mcp-in-action-track-enterprise   # Healthcare/enterprise focus
```

---

## 2. Prize Eligibility Summary

### After Phases 12-13

| Award | Amount | Eligible | Requirements Met |
|-------|--------|----------|------------------|
| Track 2: MCP in Action (1st) | $2,500 | **YES** | MCP server working |
| Modal Innovation | $2,500 | **YES** | Sandbox demo ready |
| LlamaIndex | $1,000 | **YES** | Using RAG |
| Community Choice | $1,000 | Possible | Need great demo |
| **Total Potential** | **$7,000** | | |

---

## 3. Demo Video Specification

### 3.1 Duration & Format

- **Length**: 3-4 minutes (sweet spot)
- **Format**: Screen recording + voice-over
- **Resolution**: 1080p minimum
- **Audio**: Clear narration, no background music

### 3.2 Recommended Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| OBS Studio | Screen recording | Free, cross-platform |
| Loom | Quick recording | Good for demos |
| QuickTime | Mac screen recording | Built-in |
| DaVinci Resolve | Editing | Free, professional |

### 3.3 Demo Script (4 minutes)

```markdown
## Section 1: Hook (30 seconds)

[Show Gradio UI]

"DeepBoner is an AI-powered drug repurposing research agent.
It searches peer-reviewed literature, clinical trials, and cutting-edge preprints
to find new uses for existing drugs."

"Let me show you how it works."

---

## Section 2: Core Functionality (60 seconds)

[Type query: "Can metformin treat Alzheimer's disease?"]

"When I ask about metformin for Alzheimer's, DeepBoner:
1. Searches PubMed for peer-reviewed papers
2. Queries ClinicalTrials.gov for active trials
3. Scans bioRxiv for the latest preprints"

[Show search results streaming]

"It then uses an LLM to assess the evidence quality and
synthesize findings into a structured research report."

[Show final report]

---

## Section 3: MCP Integration (60 seconds)

[Switch to Claude Desktop]

"What makes DeepBoner unique is full MCP integration.
These same tools are available to any MCP client."

[Show Claude Desktop with DeepBoner tools]

"I can ask Claude: 'Search PubMed for aspirin cancer prevention'"

[Show results appearing in Claude Desktop]

"The agent uses our MCP server to search real biomedical databases."

[Show MCP Inspector briefly]

"Here's the MCP schema - four tools exposed for any AI to use."

---

## Section 4: Modal Innovation (45 seconds)

[Run verify_sandbox.py]

"For statistical analysis, we use Modal for secure code execution."

[Show sandbox verification output]

"Notice the hostname is NOT my machine - code runs in an isolated container.
Network is blocked. The AI can't reach the internet from the sandbox."

[Run analysis demo]

"Modal executes LLM-generated statistical code safely,
returning verdicts like SUPPORTED, REFUTED, or INCONCLUSIVE."

---

## Section 5: Close (45 seconds)

[Return to Gradio UI]

"DeepBoner brings together:
- Three biomedical data sources
- MCP protocol for universal tool access
- Modal sandboxes for safe code execution
- LlamaIndex for semantic search

All in a beautiful Gradio interface."

"Check out the code on GitHub, try it on HuggingFace Spaces,
and let us know what you think."

"Thanks for watching!"

[Show links: GitHub, HuggingFace, Team names]
```

---

## 4. HuggingFace Space Configuration

### 4.1 Space README.md

```markdown
---
title: DeepBoner
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
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

# DeepBoner

AI-Powered Drug Repurposing Research Agent

## Features

- **Multi-Source Search**: PubMed, ClinicalTrials.gov, bioRxiv/medRxiv
- **MCP Integration**: Use our tools from Claude Desktop or any MCP client
- **Modal Sandbox**: Secure execution of AI-generated statistical code
- **LlamaIndex RAG**: Semantic search and evidence synthesis

## MCP Tools

Connect to our MCP server at:
```
https://your-space.hf.space/gradio_api/mcp/
```

Available tools:
- `search_pubmed` - Search peer-reviewed biomedical literature
- `search_clinical_trials` - Search ClinicalTrials.gov
- `search_biorxiv` - Search bioRxiv/medRxiv preprints
- `search_all` - Search all sources simultaneously

## Team

- The-Obstacle-Is-The-Way
- MarioAderman

## Links

- [GitHub Repository](https://github.com/The-Obstacle-Is-The-Way/DeepBoner-1)
- [Demo Video](link-to-video)
```

### 4.2 Environment Variables (Secrets)

Set in HuggingFace Space settings:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
NCBI_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

---

## 5. Social Media Post

### Twitter/X Template

```
🧬 Excited to submit DeepBoner to MCP's 1st Birthday Hackathon!

An AI agent that:
✅ Searches PubMed, ClinicalTrials.gov & bioRxiv
✅ Exposes tools via MCP protocol
✅ Runs statistical code in Modal sandboxes
✅ Uses LlamaIndex for semantic search

Try it: [HuggingFace link]
Demo: [Video link]

#MCPHackathon #AIAgents #DrugRepurposing @huggingface @AnthropicAI
```

### LinkedIn Template

```
Thrilled to share DeepBoner - our submission to MCP's 1st Birthday Hackathon!

🔬 What it does:
DeepBoner is an AI-powered drug repurposing research agent that searches
peer-reviewed literature, clinical trials, and preprints to find new uses
for existing drugs.

🛠️ Technical highlights:
• Full MCP integration - tools work with Claude Desktop
• Modal sandboxes for secure AI-generated code execution
• LlamaIndex RAG for semantic evidence search
• Three biomedical data sources in parallel

Built with PydanticAI, Gradio, and deployed on HuggingFace Spaces.

Try it: [link]
Watch the demo: [link]

#ArtificialIntelligence #Healthcare #DrugDiscovery #MCP #Hackathon
```

---

## 6. Pre-Submission Checklist

### 6.1 Code Quality

```bash
# Run all checks
make check

# Expected output:
# ✅ Linting passed (ruff)
# ✅ Type checking passed (mypy)
# ✅ All 80+ tests passed (pytest)
```

### 6.2 Documentation

- [ ] README.md updated with MCP instructions
- [ ] All demo scripts have docstrings
- [ ] Example files work end-to-end
- [ ] CLAUDE.md is current

### 6.3 Deployment Verification

```bash
# Test locally
uv run python src/app.py
# Visit http://localhost:7860

# Test MCP schema
curl http://localhost:7860/gradio_api/mcp/schema

# Test Modal (if configured)
uv run python examples/modal_demo/verify_sandbox.py
```

### 6.4 HuggingFace Space

- [ ] Space created in `MCP-1st-Birthday` organization
- [ ] Secrets configured (API keys)
- [ ] App starts without errors
- [ ] MCP endpoint accessible
- [ ] Track tag in README

---

## 7. Recording Checklist

### Before Recording

- [ ] Close unnecessary apps/notifications
- [ ] Clear browser history/tabs
- [ ] Test all demos work
- [ ] Prepare terminal windows
- [ ] Write down talking points

### During Recording

- [ ] Speak clearly and at moderate pace
- [ ] Pause briefly between sections
- [ ] Show your face? (optional, adds personality)
- [ ] Don't rush - 3-4 min is enough time

### After Recording

- [ ] Watch playback for errors
- [ ] Trim dead air at start/end
- [ ] Add title/end cards
- [ ] Export at 1080p
- [ ] Upload to YouTube/Loom

---

## 8. Submission Steps

### Step 1: Finalize Code

```bash
# Ensure clean state
git status
make check

# Push to GitHub
git push origin main

# Sync to HuggingFace
git push huggingface-upstream main
```

### Step 2: Verify HuggingFace Space

1. Visit Space URL
2. Test the chat interface
3. Test MCP endpoint: `/gradio_api/mcp/schema`
4. Verify README has track tag

### Step 3: Record Demo Video

1. Follow script from Section 3.3
2. Edit and export
3. Upload to YouTube (unlisted) or Loom
4. Copy shareable link

### Step 4: Create Social Post

1. Write post (see templates)
2. Include video link
3. Tag relevant accounts
4. Post and copy link

### Step 5: Submit

1. Ensure Space is in `MCP-1st-Birthday` org
2. Verify track tag in README
3. Submit entry (check hackathon page for form)
4. Include all links

---

## 9. Verification Commands

```bash
# 1. Full test suite
make check

# 2. Start local server
uv run python src/app.py

# 3. Verify MCP works
curl http://localhost:7860/gradio_api/mcp/schema | jq

# 4. Test with MCP Inspector
npx @anthropic/mcp-inspector http://localhost:7860/gradio_api/mcp/

# 5. Run Modal verification
uv run python examples/modal_demo/verify_sandbox.py

# 6. Run full demo
uv run python examples/orchestrator_demo/run_agent.py "metformin alzheimer"
```

---

## 10. Definition of Done

Phase 14 is **COMPLETE** when:

- [ ] Demo video recorded (3-4 min)
- [ ] Video uploaded (YouTube/Loom)
- [ ] Social media post created with link
- [ ] HuggingFace Space in `MCP-1st-Birthday` org
- [ ] Track tag in Space README
- [ ] All team members registered
- [ ] Entry submitted before deadline
- [ ] Confirmation received

---

## 11. Timeline

| Task | Time | Deadline |
|------|------|----------|
| Phase 12: MCP Server | 2-3 hours | Nov 28 |
| Phase 13: Modal Integration | 2-3 hours | Nov 29 |
| Phase 14: Demo & Submit | 2-3 hours | Nov 30 |
| **Buffer** | ~24 hours | Before 11:59 PM UTC |

---

## 12. Contact & Support

### Hackathon Resources

- Discord: `#agents-mcp-hackathon-winter25`
- HuggingFace: [MCP-1st-Birthday org](https://huggingface.co/MCP-1st-Birthday)
- MCP Docs: [modelcontextprotocol.io](https://modelcontextprotocol.io/)

### Team Communication

- Coordinate on final review
- Agree on who submits
- Celebrate when done! 🎉

---

**Good luck! Ship it with confidence.**
