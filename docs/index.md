# DeepCritical Documentation

## Medical Drug Repurposing Research Agent

AI-powered deep research system for accelerating drug repurposing discovery.

---

## Quick Links

### Architecture
- **[Overview](architecture/overview.md)** - Project overview, use case, architecture, timeline
- **[Design Patterns](architecture/design-patterns.md)** - 13 technical patterns, judge prompts, data models

### Guides
- Setup Guide (coming soon)
- User Guide (coming soon)

### Development
- Contributing (coming soon)
- API Reference (coming soon)

---

## What We're Building

**One-liner**: AI agent that searches medical literature to find existing drugs that might treat new diseases.

**Example Query**:
> "What existing drugs might help treat long COVID fatigue?"

**Output**: Research report with drug candidates, mechanisms, evidence quality, and citations.

---

## Architecture Summary

```
User Question → Research Agent (Orchestrator)
                      ↓
              Search Loop:
                → Tools (PubMed, Web Search)
                → Judge (Quality + Budget)
                → Repeat or Synthesize
                      ↓
              Research Report with Citations
```

---

## Hackathon Tracks

| Track | Status | Key Feature |
|-------|--------|-------------|
| **Gradio** | ✅ Planned | Streaming UI with progress |
| **MCP** | ✅ Planned | PubMed as MCP server |
| **Modal** | 🔄 Stretch | GPU inference option |

---

## Team

- Physician (medical domain expert) ✅
- Software engineers ✅
- AI architecture validated by multiple agents ✅

---

## Status

**Architecture Review**: PASSED (98-99/100)
**Specs**: IRONCLAD
**Next**: Implementation
