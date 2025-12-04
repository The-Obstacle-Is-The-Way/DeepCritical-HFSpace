# DeepBoner Documentation

## Sexual Health Research Agent

AI-powered deep research system for sexual wellness, reproductive health, and hormone therapy research.

---

## Quick Links

### Architecture
- **[Overview](architecture/overview.md)** - Project overview, use case, architecture
- **[Design Patterns](architecture/design-patterns.md)** - Technical patterns, data models
- **[Workflow Diagrams](workflow-diagrams.md)** - Visual architecture (Magentic v2.0)

### Implementation (Phases 1-14 ✅ COMPLETE)
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
- **[Phase 11: Europe PMC](implementation/11_phase_europepmc.md)** ✅ - Preprint search
- **[Phase 12: MCP Server](implementation/12_phase_mcp_server.md)** ✅ - Claude Desktop integration
- **[Phase 13: Modal Integration](implementation/13_phase_modal_integration.md)** ✅ - Secure code execution
- **[Phase 14: Demo Submission](implementation/14_phase_demo_submission.md)** ✅ - Hackathon submission

### Future Roadmap
- **[Overview](future-roadmap/phases/README.md)** - Planned phases 15-17
- **[Phase 15: OpenAlex](future-roadmap/phases/15_PHASE_OPENALEX.md)** - Citation network integration
- **[Phase 16: PubMed Full-text](future-roadmap/phases/16_PHASE_PUBMED_FULLTEXT.md)** - BioC API
- **[Phase 17: Rate Limiting](future-roadmap/phases/17_PHASE_RATE_LIMITING.md)** - Production hardening
- **[Deep Research Mode](future-roadmap/DEEP_RESEARCH_ROADMAP.md)** - GPT-Researcher style enhancements

### Bugs & Issues
- **[Active Bugs](bugs/ACTIVE_BUGS.md)** - Current issues and workarounds

### Decisions
- **[PR #55 Evaluation](decisions/2025-11-27-pr55-evaluation.md)** - Architecture decision record
- **[Magentic + PydanticAI](decisions/architecture-2025-11/)** - Framework architecture decisions

### Guides
- **[Deployment Guide](guides/deployment.md)** - Gradio, MCP, and Modal launch steps

### Development
- **[Testing Strategy](development/testing.md)** - Unit, Integration, and E2E testing patterns

### Brainstorming (Source Improvements)
- **[Roadmap Summary](brainstorming/00_ROADMAP_SUMMARY.md)** - Data source enhancement ideas
- **[PubMed Improvements](brainstorming/01_PUBMED_IMPROVEMENTS.md)**
- **[ClinicalTrials Improvements](brainstorming/02_CLINICALTRIALS_IMPROVEMENTS.md)**
- **[Europe PMC Improvements](brainstorming/03_EUROPEPMC_IMPROVEMENTS.md)**

---

## What We're Building

**One-liner**: AI agent that searches medical literature to find evidence for sexual health research questions.

**Example Queries**:
> "What drugs improve female libido post-menopause?"
> "Evidence for testosterone therapy in women with HSDD?"
> "Clinical trials for erectile dysfunction alternatives to PDE5 inhibitors?"

**Output**: Research report with drug candidates, mechanisms, evidence quality, and citations.

---

## Architecture Summary

```
User Question → Research Agent (Orchestrator)
                      ↓
              Search Loop:
                → Tools (PubMed, ClinicalTrials, Europe PMC)
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
| **Multi-Source Search** | ✅ Complete | PubMed, ClinicalTrials, Europe PMC |

---

## Status

| Phase | Status |
|-------|--------|
| Phases 1-14 | ✅ COMPLETE |

**Tests**: 318 passing, 0 warnings
**Known Issues**: See [Active Bugs](bugs/ACTIVE_BUGS.md)
