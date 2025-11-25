# DeepCritical Documentation

## Medical Drug Repurposing Research Agent

AI-powered deep research system for accelerating drug repurposing discovery.

---

## Quick Links

### Architecture
- **[Overview](architecture/overview.md)** - Project overview, use case, architecture, timeline
- **[Design Patterns](architecture/design-patterns.md)** - 17 technical patterns, reference repos, judge prompts, data models

### Implementation (Start Here!)
- **[Roadmap](implementation/roadmap.md)** - Phased execution plan with TDD
- **[Phase 1: Foundation](implementation/01_phase_foundation.md)** - Tooling, config, first tests
- **[Phase 2: Search](implementation/02_phase_search.md)** - PubMed + DuckDuckGo
- **[Phase 3: Judge](implementation/03_phase_judge.md)** - LLM evidence assessment
- **[Phase 4: UI](implementation/04_phase_ui.md)** - Orchestrator + Gradio + Deploy

### Guides
- [Setup Guide](guides/setup.md) (coming soon)
- **[Deployment Guide](guides/deployment.md)** - Gradio, MCP, and Modal launch steps

### Development
- **[Testing Strategy](development/testing.md)** - Unit, Integration, and E2E testing patterns
- [Contributing](development/contributing.md) (coming soon)


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
