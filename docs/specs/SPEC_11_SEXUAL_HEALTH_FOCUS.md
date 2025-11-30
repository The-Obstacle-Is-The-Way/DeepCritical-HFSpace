# SPEC_11: Sexual Health Research Specialist (Final Polish)

**Status**: APPROVED
**Priority**: P0 (Critical Fix)
**Effort**: Low (Cleanup & Polish)
**Related Issues**: #75, #89

## 1. Executive Summary

DeepBoner is **exclusively** a Sexual Health Research Agent. The codebase is currently in a transitional state where "General" and "Drug Repurposing" modes were architecturally removed, but significant artifacts (docstrings, default arguments, variable names, and examples) remain.

This specification dictates the **complete eradication** of non-sexual-health concepts from the codebase to ensure a consistent, focused, and professional product identity.

## 2. The Rules of Engagement

1.  **No "General" Defaults**: The string literal `"general"` shall not exist as a default value for any `domain` parameter.
2.  **No "Drug Repurposing" References**: Terms like "metformin", "alzheimer", "cancer", "aspirin" in examples must be replaced with sexual health examples.
3.  **Single Source of Truth**: `src.config.domain.ResearchDomain.SEXUAL_HEALTH` is the *only* valid domain.
4.  **Ironclad Tests**: Tests must use sexual health queries (e.g., "libido", "testosterone", "PDE5") to ensure the domain logic is actually exercising the production paths.

## 3. Implementation Plan

### 3.1. Code Cleanup (`src/`)

#### `src/app.py`
- **Logic Fix**: Change `domain_str = domain or "general"` to `domain_str = domain or "sexual_health"`.
- **Signature Fix**: Change `domain: str = "general"` to `domain: str = "sexual_health"`.
- **Docstring Fix**: Remove `(e.g., "general", "sexual_health")`.

#### `src/mcp_tools.py`
- **Signature Fix**: Update `search_pubmed` and `search_all_sources` to default `domain="sexual_health"`.
- **Docstring Fix**: Update examples from "metformin alzheimer" to "testosterone libido".
- **Argument Description**: Remove `(general, drug_repurposing, sexual_health)` list.

#### `src/tools/*.py`
- **`clinicaltrials.py`, `query_utils.py`, `tools.py`**: Replace all "metformin/alzheimer" example strings with sexual health examples.

#### `src/config/domain.py`
- **Comment Fix**: Remove `# Get default (general) config`.

### 3.2. Test Suite Alignment (`tests/`)

#### `tests/unit/agent_factory/test_judges.py`
- Replace `metformin alzheimer` test queries with `sildenafil efficacy`.

#### `tests/unit/tools/test_query_utils.py`
- Ensure synonym expansion tests use relevant terms (or generic ones that don't imply a different domain).

#### `tests/unit/mcp/test_mcp_tools_domain.py`
- Verify defaults are "sexual_health", not "general".

## 4. Verification Checklist

- [ ] **Grep Audit**: `grep -r "general" src/` should return zero results where it refers to a domain default.
- [ ] **Grep Audit**: `grep -r "metformin" src/` should return zero results.
- [ ] **Functionality**: `src/app.py` runs without crashing when `domain` is `None` (defaults to sexual_health).
- [ ] **Tests**: All 237+ tests pass.

## 5. Success State

When this spec is implemented, a developer reading the code should see **zero evidence** that this agent was ever intended for anything other than Sexual Health research.