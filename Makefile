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

check: lint typecheck test
	@echo "All checks passed!"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
