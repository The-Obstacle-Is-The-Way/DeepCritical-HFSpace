# SPEC_10: Domain-Agnostic Refactor

**Status**: DRAFT
**Priority**: P1
**Effort**: Medium (2-3 hours)
**Related Issues**: #75, #76

## Problem Statement

The codebase has "drug repurposing" hardcoded in **16 locations** (originally identified 15, plus 1 found in audit):

```
src/prompts/report.py:11      - SYSTEM_PROMPT
src/prompts/judge.py:5        - SYSTEM_PROMPT
src/prompts/judge.py:140      - Evidence scoring prompt (inside format_user_prompt)
src/prompts/hypothesis.py:11  - SYSTEM_PROMPT
src/orchestrators/simple.py:476   - Report header
src/orchestrators/simple.py:564   - Report header
src/orchestrators/advanced.py:159 - Task prompt
src/agents/magentic_agents.py:33  - Agent description
src/agents/magentic_agents.py:108 - Agent description
src/agents/search_agent.py:31     - Tool description
src/agents/tools.py:85            - Tool docstring
src/mcp_tools.py:27               - Example query
src/mcp_tools.py:116              - Docstring
src/mcp_tools.py:164              - Function docstring
src/mcp_tools.py:167              - Docstring
src/agent_factory/judges.py:21    - Imports format_user_prompt (needs update)
```

This violates:
- **DRY** - Same concept repeated 15+ times
- **Open/Closed** - Can't add domains without modifying multiple files
- **Flexibility** - Agent is locked to one domain

## Solution: Centralized Domain Configuration

### 1. Create Domain Config Module

**File**: `src/config/domain.py`

```python
"""Centralized domain configuration for research agents.

This module defines research domains and their associated prompts,
allowing the agent to operate in domain-agnostic or domain-specific modes.

Usage:
    from src.config.domain import get_domain_config, ResearchDomain

    # Get default (general) config
    config = get_domain_config()

    # Get specific domain
    config = get_domain_config(ResearchDomain.SEXUAL_HEALTH)

    # Use in prompts
    system_prompt = config.judge_system_prompt
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel


class ResearchDomain(str, Enum):
    """Available research domains."""

    GENERAL = "general"
    DRUG_REPURPOSING = "drug_repurposing"
    SEXUAL_HEALTH = "sexual_health"


class DomainConfig(BaseModel):
    """Configuration for a research domain.

    Contains all domain-specific text used across the codebase,
    ensuring consistency and single-source-of-truth.
    """

    # Identity
    name: str
    description: str

    # Report generation
    report_title: str
    report_focus: str

    # Judge prompts
    judge_system_prompt: str
    judge_scoring_prompt: str

    # Hypothesis prompts
    hypothesis_system_prompt: str

    # Report writer prompts
    report_system_prompt: str

    # Search context
    search_description: str
    search_example_query: str

    # Agent descriptions (for Magentic mode)
    search_agent_description: str
    hypothesis_agent_description: str


# ─────────────────────────────────────────────────────────────────
# Domain Definitions
# ─────────────────────────────────────────────────────────────────

GENERAL_CONFIG = DomainConfig(
    name="General Research",
    description="General-purpose biomedical research agent",

    report_title="## Research Analysis",
    report_focus="comprehensive research synthesis",

    judge_system_prompt="""You are an expert research judge.
Your role is to evaluate evidence quality, assess relevance to the research query,
and determine if sufficient evidence exists to synthesize findings.""",

    judge_scoring_prompt="""Score this evidence for research relevance.
Provide ONLY scores and extracted data.""",

    hypothesis_system_prompt="""You are a biomedical research scientist.
Your role is to generate evidence-based hypotheses from the literature,
identifying key mechanisms, targets, and potential therapeutic implications.""",

    report_system_prompt="""You are a scientific writer specializing in research reports.
Your role is to synthesize evidence into clear, well-structured reports with
proper citations and evidence-based conclusions.""",

    search_description="Searches biomedical literature for relevant evidence",
    search_example_query="metformin aging mechanisms",

    search_agent_description="Searches PubMed, ClinicalTrials.gov, and Europe PMC for evidence",
    hypothesis_agent_description="Generates mechanistic hypotheses from evidence",
)

DRUG_REPURPOSING_CONFIG = DomainConfig(
    name="Drug Repurposing",
    description="Drug repurposing research specialist",

    report_title="## Drug Repurposing Analysis",
    report_focus="drug repurposing opportunities",

    judge_system_prompt="""You are an expert drug repurposing research judge.
Your role is to evaluate evidence for drug repurposing potential, assess
mechanism plausibility, and determine if compounds warrant further investigation.""",

    judge_scoring_prompt="""Score this evidence for drug repurposing potential.
Provide ONLY scores and extracted data.""",

    hypothesis_system_prompt="""You are a biomedical research scientist specializing in drug repurposing.
Your role is to generate mechanistic hypotheses for how existing drugs might
treat new indications, based on shared pathways and targets.""",

    report_system_prompt="""You are a scientific writer specializing in drug repurposing research reports.
Your role is to synthesize evidence into actionable drug repurposing recommendations
with clear mechanistic rationale and clinical translation potential.""",

    search_description="Searches biomedical literature for drug repurposing evidence",
    search_example_query="metformin alzheimer repurposing",

    search_agent_description="Searches PubMed for drug repurposing evidence",
    hypothesis_agent_description="Generates mechanistic hypotheses for drug repurposing",
)

SEXUAL_HEALTH_CONFIG = DomainConfig(
    name="Sexual Health Research",
    description="Sexual health and wellness research specialist",

    report_title="## Sexual Health Analysis",
    report_focus="sexual health and wellness interventions",

    judge_system_prompt="""You are an expert sexual health research judge.
Your role is to evaluate evidence for sexual health interventions, assess
efficacy and safety data, and determine clinical applicability.""",

    judge_scoring_prompt="""Score this evidence for sexual health relevance.
Provide ONLY scores and extracted data.""",

    hypothesis_system_prompt="""You are a biomedical research scientist specializing in sexual health.
Your role is to generate evidence-based hypotheses for sexual health interventions,
identifying mechanisms of action and potential therapeutic applications.""",

    report_system_prompt="""You are a scientific writer specializing in sexual health research reports.
Your role is to synthesize evidence into clear recommendations for sexual health
interventions with proper safety considerations.""",

    search_description="Searches biomedical literature for sexual health evidence",
    search_example_query="testosterone therapy female libido",

    search_agent_description="Searches PubMed for sexual health evidence",
    hypothesis_agent_description="Generates hypotheses for sexual health interventions",
)

# ─────────────────────────────────────────────────────────────────
# Domain Registry
# ─────────────────────────────────────────────────────────────────

DOMAIN_CONFIGS: dict[ResearchDomain, DomainConfig] = {
    ResearchDomain.GENERAL: GENERAL_CONFIG,
    ResearchDomain.DRUG_REPURPOSING: DRUG_REPURPOSING_CONFIG,
    ResearchDomain.SEXUAL_HEALTH: SEXUAL_HEALTH_CONFIG,
}

# Default domain
DEFAULT_DOMAIN = ResearchDomain.GENERAL


def get_domain_config(domain: ResearchDomain | str | None = None) -> DomainConfig:
    """Get configuration for a research domain.

    Args:
        domain: The research domain. Defaults to GENERAL if None.

    Returns:
        DomainConfig for the specified domain.
    """
    if domain is None:
        domain = DEFAULT_DOMAIN
    
    if isinstance(domain, str):
        try:
            domain = ResearchDomain(domain)
        except ValueError:
            domain = DEFAULT_DOMAIN

    return DOMAIN_CONFIGS[domain]
```

### 2. Update Settings to Include Domain

**File**: `src/utils/config.py` (add to Settings class)

```python
from src.config.domain import ResearchDomain

class Settings(BaseSettings):
    # ... existing fields ...

    # Domain configuration
    research_domain: ResearchDomain = ResearchDomain.GENERAL
```

### 3. Update All Hardcoded Locations

#### 3.1 Prompts Module

**`src/prompts/report.py`**:
```python
from src.config.domain import get_domain_config

def get_system_prompt(domain=None):
    config = get_domain_config(domain)
    return config.report_system_prompt

# Keep SYSTEM_PROMPT for backwards compatibility (uses default)
SYSTEM_PROMPT = get_system_prompt()
```

**`src/prompts/judge.py`**:
```python
from src.config.domain import get_domain_config, ResearchDomain

def get_system_prompt(domain=None):
    config = get_domain_config(domain)
    return config.judge_system_prompt

def format_user_prompt(
    question: str,
    evidence: list[Evidence],
    iteration: int = 0,
    max_iterations: int = 10,
    total_evidence_count: int | None = None,
    domain: ResearchDomain | None = None,  # NEW ARGUMENT
) -> str:
    config = get_domain_config(domain)
    # ... existing logic ...
    
    # Inside f-string:
    return f"""...
{config.judge_scoring_prompt}
DO NOT decide "synthesize" vs "continue" - that decision is made by the system.
...
"""

SYSTEM_PROMPT = get_system_prompt()
```

**`src/prompts/hypothesis.py`**:
```python
from src.config.domain import get_domain_config

def get_system_prompt(domain=None):
    config = get_domain_config(domain)
    return config.hypothesis_system_prompt

SYSTEM_PROMPT = get_system_prompt()
```

#### 3.2 Judge Factory

**`src/agent_factory/judges.py`**:
```python
from src.config.domain import ResearchDomain

class JudgeHandler:
    def __init__(self, model: Any = None, domain: ResearchDomain | None = None) -> None:
        self.model = model or get_model()
        self.domain = domain  # Store domain
        # ...

    async def assess(self, ...):
        # ...
        if evidence:
            user_prompt = format_user_prompt(
                ...,
                domain=self.domain  # Pass domain
            )
```

#### 3.3 Orchestrators

**`src/orchestrators/simple.py`**:
```python
from src.config.domain import get_domain_config

class SimpleOrchestrator:
    def __init__(self, domain=None, ...):
        self.domain = domain
        self.domain_config = get_domain_config(domain)
        
        # Pass domain to JudgeHandler
        self.judge = JudgeHandler(domain=domain)

    def _format_report(self, ...):
        return f"""{self.domain_config.report_title}
Query: {query}
...
"""
```

**`src/orchestrators/advanced.py`**:
```python
from src.config.domain import get_domain_config

async def run_research(..., domain=None):
    config = get_domain_config(domain)
    task = f"""Research {config.report_focus} for: {query}
    ...
    """
```

#### 3.4 Agents

**`src/agents/magentic_agents.py`**:
```python
from src.config.domain import get_domain_config

def create_search_agent(domain=None):
    config = get_domain_config(domain)
    return Agent(
        description=config.search_agent_description,
        ...
    )
```

**`src/agents/search_agent.py`** and **`src/agents/tools.py`**:
Similar pattern - inject domain config.

#### 3.5 MCP Tools

**`src/mcp_tools.py`**:
```python
from src.config.domain import get_domain_config, ResearchDomain

@mcp.tool
async def search_pubmed(query: str, domain: str = "general"):
    """Search PubMed for biomedical literature.

    Args:
        query: Search query (e.g., "metformin alzheimer")
        domain: Research domain (general, drug_repurposing, sexual_health)
    """
    config = get_domain_config(ResearchDomain(domain))
    # Use config.search_description in responses
```

### 4. Update Gradio UI

**`src/app.py`** - Add domain selector:

```python
from src.config.domain import ResearchDomain, DOMAIN_CONFIGS

domain_dropdown = gr.Dropdown(
    choices=[d.value for d in ResearchDomain],
    value="general",
    label="Research Domain",
    info="Select research focus area"
)
```

## Implementation Checklist

- [ ] Create `src/config/domain.py` with DomainConfig
- [ ] Add `research_domain` to Settings
- [ ] Update `src/prompts/report.py`
- [ ] Update `src/prompts/judge.py` (Add domain arg to `format_user_prompt`)
- [ ] Update `src/prompts/hypothesis.py`
- [ ] Update `src/agent_factory/judges.py` (Pass domain to `format_user_prompt`)
- [ ] Update `src/orchestrators/simple.py` (Pass domain to `JudgeHandler`)
- [ ] Update `src/orchestrators/advanced.py`
- [ ] Update `src/agents/magentic_agents.py`
- [ ] Update `src/agents/search_agent.py`
- [ ] Update `src/agents/tools.py`
- [ ] Update `src/mcp_tools.py`
- [ ] Add domain selector to Gradio UI
- [ ] **Update Tests**: `tests/e2e/test_simple_mode.py` contains hardcoded "Drug Repurposing" assertions that will fail with default "General" domain.

## Testing Strategy

### Unit Tests

```python
# tests/unit/config/test_domain.py

def test_get_domain_config_default():
    config = get_domain_config()
    assert config.name == "General Research"

def test_get_domain_config_drug_repurposing():
    config = get_domain_config(ResearchDomain.DRUG_REPURPOSING)
    assert "drug repurposing" in config.judge_system_prompt.lower()

def test_all_domains_have_required_fields():
    for domain in ResearchDomain:
        config = get_domain_config(domain)
        assert config.report_title
        assert config.judge_system_prompt
        assert config.hypothesis_system_prompt
```

### Integration Tests

```python
# tests/integration/test_domain_switching.py

@pytest.mark.integration
async def test_simple_mode_respects_domain():
    result = await run_simple_mode(
        "metformin aging",
        domain=ResearchDomain.GENERAL
    )
    assert "## Research Analysis" in result

    result = await run_simple_mode(
        "metformin aging",
        domain=ResearchDomain.DRUG_REPURPOSING
    )
    assert "## Drug Repurposing Analysis" in result
```

## Migration Path

1. **Phase 1**: Create domain config, add to Settings (no breaking changes)
2. **Phase 2**: Update prompts module to use config (backwards compatible)
3. **Phase 3**: Update `JudgeHandler` and `format_user_prompt` (requires careful threading of domain)
4. **Phase 4**: Update orchestrators and agents
5. **Phase 5**: Update UI with domain selector and Fix Tests

## Success Criteria

- [ ] Zero hardcoded "drug repurposing" strings in `src/` (except `domain.py`)
- [ ] All existing tests pass (after updates)
- [ ] New domain can be added by only modifying `domain.py`
- [ ] Default behavior is "General Research"