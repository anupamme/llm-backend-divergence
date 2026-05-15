.PHONY: setup test lint format clean

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check divergence/
	uv run ruff format --check divergence/
	uv run mypy --strict divergence/

format:
	uv run ruff format divergence/
	uv run ruff check --fix divergence/

clean:
	rm -rf .venv/ .mypy_cache/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
