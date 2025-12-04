.PHONY: install test lint format typecheck check clean all cov cov-html

# Default target
all: check

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest tests/unit/ -v

# Coverage aliases
cov: test-cov
test-cov:
	uv run pytest --cov=src --cov-report=term-missing

cov-html:
	uv run pytest --cov=src --cov-report=html
	@echo "Coverage report: open htmlcov/index.html"

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

# Run all checks (lint, typecheck, test)
check: lint typecheck test

# Smoke tests - run against real APIs (slow, not for CI)
smoke-free:
	@echo "Running Free Tier smoke test..."
	uv run python -m pytest tests/e2e/test_smoke.py::test_free_tier_synthesis -v -s

smoke-paid:
	@echo "Running Paid Tier smoke test (requires OPENAI_API_KEY)..."
	uv run python -m pytest tests/e2e/test_smoke.py::test_paid_tier_synthesis -v -s

smoke: smoke-free  # Default to free tier

# Clean up cache and artifacts

