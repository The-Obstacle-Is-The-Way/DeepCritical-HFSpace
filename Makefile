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
