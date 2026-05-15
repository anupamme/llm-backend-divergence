.PHONY: setup setup-llamacpp test test-slow lint format clean

setup:
	uv sync

setup-llamacpp:
	CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++" uv pip install --python .venv/bin/python "llama-cpp-python>=0.3"

test:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest -m slow

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
