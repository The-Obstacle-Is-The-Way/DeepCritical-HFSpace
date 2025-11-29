# DeepBoner Documentation

## Medical Drug Repurposing Research Agent

AI-powered deep research system for accelerating drug repurposing discovery.

---

## Quick Links

### Architecture
- **[Overview](architecture/overview.md)** - Project overview, use case, architecture
- **[Design Patterns](architecture/design-patterns.md)** - Technical patterns, data models

### Implementation
- **[Roadmap](implementation/roadmap.md)** - Phased execution plan with TDD
- **[Phase 1: Foundation](implementation/01_phase_foundation.md)** ✅ - Tooling, config, first tests
- **[Phase 2: Search](implementation/02_phase_search.md)** ✅ - PubMed search
- **[Phase 3: Judge](implementation/03_phase_judge.md)** ✅ - LLM evidence assessment
- **[Phase 4: UI](implementation/04_phase_ui.md)** ✅ - Orchestrator + Gradio
- **[Phase 5: Magentic](implementation/05_phase_magentic.md)** ✅ - Multi-agent orchestration
- **[Phase 6: Embeddings](implementation/06_phase_embeddings.md)** ✅ - Semantic search + dedup
- **[Phase 7: Hypothesis](implementation/07_phase_hypothesis.md)** ✅ - Mechanistic reasoning
- **[Phase 8: Report](implementation/08_phase_report.md)** ✅ - Structured scientific reports
- **[Phase 9: Source Cleanup](implementation/09_phase_source_cleanup.md)** ✅ - Remove DuckDuckGo
- **[Phase 10: ClinicalTrials](implementation/10_phase_clinicaltrials.md)** ✅ - Clinical trials API
- **[Phase 11: bioRxiv](implementation/11_phase_biorxiv.md)** ✅ - Preprint search
- **[Phase 12: MCP Server](implementation/12_phase_mcp_server.md)** ✅ - Claude Desktop integration
- **[Phase 13: Modal Integration](implementation/13_phase_modal_integration.md)** ✅ - Secure code execution
- **[Phase 14: Demo Submission](implementation/14_phase_demo_submission.md)** ✅ - Hackathon submission

### Guides
- **[Deployment Guide](guides/deployment.md)** - Gradio, MCP, and Modal launch steps

### Development
- **[Testing Strategy](development/testing.md)** - Unit, Integration, and E2E testing patterns

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
                → Tools (PubMed, ClinicalTrials, bioRxiv)
                → Judge (Quality + Budget)
                → Repeat or Synthesize
                      ↓
              Research Report with Citations
```

---

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Gradio UI** | ✅ Complete | Streaming chat interface |
| **MCP Server** | ✅ Complete | Tools accessible from Claude Desktop |
| **Modal Sandbox** | ✅ Complete | Secure statistical analysis |
| **Multi-Source Search** | ✅ Complete | PubMed, ClinicalTrials, bioRxiv |

---

## Team

- The-Obstacle-Is-The-Way
- MarioAderman
- Josephrp

---

## Status

| Phase | Status |
|-------|--------|
| Phases 1-14 | ✅ COMPLETE |

**Test Coverage**: 65% (96 tests passing)
**Architecture Review**: PASSED (98-99/100)
