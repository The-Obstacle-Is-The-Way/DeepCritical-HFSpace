# Phase 1 Implementation Spec: Foundation & Tooling

**Goal**: Establish a "Gucci Banger" development environment using 2025 best practices.
**Philosophy**: "If the build isn't solid, the agent won't be."

---

## 1. Prerequisites

Before starting, ensure these are installed:

```bash
# Install uv (Rust-based package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version  # Should be >= 0.4.0
```

---

## 2. Project Initialization

```bash
# From project root
uv init --name deepcritical
uv python install 3.11  # Pin Python version
```

---

## 3. The Tooling Stack (Exact Dependencies)

### `pyproject.toml` (Complete, Copy-Paste Ready)

```toml
[project]
name = "deepcritical"
version = "0.1.0"
description = "AI-Native Drug Repurposing Research Agent"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    # Core
    "pydantic>=2.7",
    "pydantic-settings>=2.2",      # For BaseSettings (config)
    "pydantic-ai>=0.0.16",          # Agent framework

    # HTTP & Parsing
    "httpx>=0.27",                   # Async HTTP client
    "beautifulsoup4>=4.12",          # HTML parsing
    "xmltodict>=0.13",               # PubMed XML -> dict

    # Search
    "duckduckgo-search>=6.0",        # Free web search

    # UI
    "gradio>=5.0",                   # Chat interface

    # Utils
    "python-dotenv>=1.0",            # .env loading
    "tenacity>=8.2",                 # Retry logic
    "structlog>=24.1",               # Structured logging
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-sugar>=1.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.12",
    "respx>=0.21",                   # Mock httpx requests

    # Quality
    "ruff>=0.4.0",
    "mypy>=1.10",
    "pre-commit>=3.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

# ============== RUFF CONFIG ==============
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "B",    # flake8-bugbear
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "PL",   # pylint
    "RUF",  # ruff-specific
]
ignore = [
    "PLR0913",  # Too many arguments (agents need many params)
]

[tool.ruff.lint.isort]
known-first-party = ["src"]

# ============== MYPY CONFIG ==============
[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
disallow_untyped_defs = true
warn_return_any = true
warn_unused_ignores = true

# ============== PYTEST CONFIG ==============
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
markers = [
    "unit: Unit tests (mocked)",
    "integration: Integration tests (real APIs)",
    "slow: Slow tests",
]

# ============== COVERAGE CONFIG ==============
[tool.coverage.run]
source = ["src"]
omit = ["*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## 4. Directory Structure (Maintainer's Structure)

```bash
# Execute these commands to create the directory structure
mkdir -p src/utils
mkdir -p src/tools
mkdir -p src/prompts
mkdir -p src/agent_factory
mkdir -p src/middleware
mkdir -p src/database_services
mkdir -p src/retrieval_factory
mkdir -p tests/unit/tools
mkdir -p tests/unit/agent_factory
mkdir -p tests/unit/utils
mkdir -p tests/integration

# Create __init__.py files (required for imports)
touch src/__init__.py
touch src/utils/__init__.py
touch src/tools/__init__.py
touch src/prompts/__init__.py
touch src/agent_factory/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/unit/tools/__init__.py
touch tests/unit/agent_factory/__init__.py
touch tests/unit/utils/__init__.py
touch tests/integration/__init__.py
```

### Final Structure:

```
src/
├── __init__.py
├── app.py                      # Entry point (Gradio UI)
├── orchestrator.py             # Agent loop
├── agent_factory/              # Agent creation and judges
│   ├── __init__.py
│   ├── agents.py
│   └── judges.py
├── tools/                      # Search tools
│   ├── __init__.py
│   ├── pubmed.py
│   ├── websearch.py
│   └── search_handler.py
├── prompts/                    # Prompt templates
│   ├── __init__.py
│   └── judge.py
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── models.py
│   ├── dataloaders.py
│   └── parsers.py
├── middleware/                 # (Future)
├── database_services/          # (Future)
└── retrieval_factory/          # (Future)

tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── __init__.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── test_pubmed.py
│   │   ├── test_websearch.py
│   │   └── test_search_handler.py
│   ├── agent_factory/
│   │   ├── __init__.py
│   │   └── test_judges.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── test_config.py
│   └── test_orchestrator.py
└── integration/
    ├── __init__.py
    └── test_pubmed_live.py
```

---

## 5. Configuration Files

### `.env.example` (Copy to `.env` and fill)

```bash
# LLM Provider (choose one)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: PubMed API key (higher rate limits)
NCBI_API_KEY=your-ncbi-key-here

# Optional: For HuggingFace deployment
HF_TOKEN=hf_your-token-here

# Agent Config
MAX_ITERATIONS=10
LOG_LEVEL=INFO
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.7
          - pydantic-settings>=2.2
        args: [--ignore-missing-imports]
```

### `tests/conftest.py` (Pytest Fixtures)

```python
"""Shared pytest fixtures for all tests."""
import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def mock_httpx_client(mocker):
    """Mock httpx.AsyncClient for API tests."""
    mock = mocker.patch("httpx.AsyncClient")
    mock.return_value.__aenter__ = AsyncMock(return_value=mock.return_value)
    mock.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_llm_response():
    """Factory fixture for mocking LLM responses."""
    def _mock(content: str):
        return AsyncMock(return_value=content)
    return _mock


@pytest.fixture
def sample_evidence():
    """Sample Evidence objects for testing."""
    from src.utils.models import Evidence, Citation
    return [
        Evidence(
            content="Metformin shows promise in Alzheimer's...",
            citation=Citation(
                source="pubmed",
                title="Metformin and Alzheimer's Disease",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                date="2024-01-15"
            ),
            relevance=0.85
        )
    ]
```

---

## 6. Core Utilities Implementation

### `src/utils/config.py`

```python
"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal
import structlog


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Which LLM provider to use"
    )
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model")

    # PubMed Configuration
    ncbi_api_key: str | None = Field(default=None, description="NCBI API key for higher rate limits")

    # Agent Configuration
    max_iterations: int = Field(default=10, ge=1, le=50)
    search_timeout: int = Field(default=30, description="Seconds to wait for search")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        else:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key


def get_settings() -> Settings:
    """Factory function to get settings (allows mocking in tests)."""
    return Settings()


def configure_logging(settings: Settings) -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# Singleton for easy import
settings = get_settings()
```

### `src/utils/exceptions.py`

```python
"""Custom exceptions for DeepCritical."""


class DeepCriticalError(Exception):
    """Base exception for all DeepCritical errors."""
    pass


class SearchError(DeepCriticalError):
    """Raised when a search operation fails."""
    pass


class JudgeError(DeepCriticalError):
    """Raised when the judge fails to assess evidence."""
    pass


class ConfigurationError(DeepCriticalError):
    """Raised when configuration is invalid."""
    pass


class RateLimitError(SearchError):
    """Raised when we hit API rate limits."""
    pass
```

---

## 7. TDD Workflow: First Test

### `tests/unit/utils/test_config.py`

```python
"""Unit tests for configuration loading."""
import pytest
from unittest.mock import patch
import os


class TestSettings:
    """Tests for Settings class."""

    def test_default_max_iterations(self):
        """Settings should have default max_iterations of 10."""
        from src.utils.config import Settings

        # Clear any env vars
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.max_iterations == 10

    def test_max_iterations_from_env(self):
        """Settings should read MAX_ITERATIONS from env."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {"MAX_ITERATIONS": "25"}):
            settings = Settings()
            assert settings.max_iterations == 25

    def test_invalid_max_iterations_raises(self):
        """Settings should reject invalid max_iterations."""
        from src.utils.config import Settings
        from pydantic import ValidationError

        with patch.dict(os.environ, {"MAX_ITERATIONS": "100"}):
            with pytest.raises(ValidationError):
                Settings()  # 100 > 50 (max)

    def test_get_api_key_openai(self):
        """get_api_key should return OpenAI key when provider is openai."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key"
        }):
            settings = Settings()
            assert settings.get_api_key() == "sk-test-key"

    def test_get_api_key_missing_raises(self):
        """get_api_key should raise when key is not set."""
        from src.utils.config import Settings

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                settings.get_api_key()
```

---

## 8. Makefile (Developer Experience)

Create a `Makefile` for standard devex commands:

```makefile
.PHONY: install test lint format typecheck check clean

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest tests/unit/ -v

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

check: lint typecheck test
	@echo "All checks passed!"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

---

## 9. Execution Commands

```bash
# Install all dependencies
uv sync --all-extras

# Run tests (should pass after implementing config.py)
uv run pytest tests/unit/utils/test_config.py -v

# Run full test suite with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run linting
uv run ruff check src tests
uv run ruff format src tests

# Run type checking
uv run mypy src

# Set up pre-commit hooks
uv run pre-commit install
```

---

## 10. Implementation Checklist

- [ ] Install `uv` and verify version
- [ ] Run `uv init --name deepcritical`
- [ ] Create `pyproject.toml` (copy from above)
- [ ] Create directory structure (run mkdir commands)
- [ ] Create `.env.example` and `.env`
- [ ] Create `.pre-commit-config.yaml`
- [ ] Create `Makefile` (copy from above)
- [ ] Create `tests/conftest.py`
- [ ] Implement `src/utils/config.py`
- [ ] Implement `src/utils/exceptions.py`
- [ ] Write tests in `tests/unit/utils/test_config.py`
- [ ] Run `make install`
- [ ] Run `make check` — **ALL CHECKS MUST PASS**
- [ ] Commit: `git commit -m "feat: phase 1 foundation complete"`

---

## 11. Definition of Done

Phase 1 is **COMPLETE** when:

1. `uv run pytest` passes with 100% of tests green
2. `uv run ruff check src tests` has 0 errors
3. `uv run mypy src` has 0 errors
4. Pre-commit hooks are installed and working
5. `from src.utils.config import settings` works in Python REPL

**Proceed to Phase 2 ONLY after all checkboxes are complete.**
