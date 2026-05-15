"""Tests for the llama.cpp backends."""

from __future__ import annotations

import os

import pytest

from divergence.backends import (
    Backend,
    BackendMetadata,
    Hardware,
    InferenceResult,
    LlamaCppQ4KMBackend,
    LlamaCppQ8Backend,
    ScoringResult,
)

pytestmark = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("llama_cpp"),
    reason="llama-cpp-python not installed",
)


class TestLlamaCppQ8Unit:
    def test_is_subclass_of_backend(self) -> None:
        assert issubclass(LlamaCppQ8Backend, Backend)

    def test_name(self) -> None:
        backend = LlamaCppQ8Backend()
        assert backend.name == "llamacpp-q8"

    def test_metadata(self) -> None:
        backend = LlamaCppQ8Backend()
        meta = backend.metadata
        assert isinstance(meta, BackendMetadata)
        assert meta.quantization == "Q8_0"
        assert meta.dtype == "float16"
        assert meta.hardware == Hardware.METAL
        assert meta.extra == {"n_gpu_layers": -1}

    def test_generate_raises_if_not_loaded(self) -> None:
        backend = LlamaCppQ8Backend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)

    def test_score_raises_if_not_loaded(self) -> None:
        backend = LlamaCppQ8Backend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.score("x", "y")


class TestLlamaCppQ4KMUnit:
    def test_is_subclass_of_backend(self) -> None:
        assert issubclass(LlamaCppQ4KMBackend, Backend)

    def test_name(self) -> None:
        backend = LlamaCppQ4KMBackend()
        assert backend.name == "llamacpp-q4km"

    def test_metadata(self) -> None:
        backend = LlamaCppQ4KMBackend()
        meta = backend.metadata
        assert isinstance(meta, BackendMetadata)
        assert meta.quantization == "Q4_K_M"
        assert meta.dtype == "float16"
        assert meta.hardware == Hardware.METAL
        assert meta.extra == {"n_gpu_layers": -1}

    def test_generate_raises_if_not_loaded(self) -> None:
        backend = LlamaCppQ4KMBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)

    def test_score_raises_if_not_loaded(self) -> None:
        backend = LlamaCppQ4KMBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.score("x", "y")


@pytest.mark.slow
class TestLlamaCppQ4KMIntegration:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"

    @pytest.fixture(scope="class")
    def backend(self) -> LlamaCppQ4KMBackend:
        b = LlamaCppQ4KMBackend()
        b.load(self.MODEL_ID)
        yield b  # type: ignore[misc]
        b.unload()

    def test_generate_produces_output(self, backend: LlamaCppQ4KMBackend) -> None:
        result = backend.generate(
            "The capital of France is",
            max_tokens=32,
            temperature=0.0,
            seed=42,
        )
        assert isinstance(result, InferenceResult)
        assert result.completion != ""
        assert len(result.token_ids) > 0
        assert len(result.token_ids) <= 32
        assert result.ttft_ms < 30_000
        assert result.backend_name == "llamacpp-q4km"
        assert result.finish_reason in ("stop", "length")

    def test_determinism_with_seed(self, backend: LlamaCppQ4KMBackend) -> None:
        r1 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        r2 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        assert r1.token_ids == r2.token_ids
        assert r1.completion == r2.completion

    def test_score_returns_logprobs(self, backend: LlamaCppQ4KMBackend) -> None:
        result = backend.score("The capital of France is", " Paris")
        assert isinstance(result, ScoringResult)
        assert len(result.logprobs) == len(result.token_ids)
        assert len(result.token_ids) > 0
        assert all(lp <= 0.0 for lp in result.logprobs)
        assert result.backend_name == "llamacpp-q4km"

    def test_peak_memory_under_6gb(self, backend: LlamaCppQ4KMBackend) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        backend.generate(
            "Write a short essay about AI.",
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
        rss_gb = process.memory_info().rss / (1024**3)
        assert rss_gb < 6.0, f"Peak RSS was {rss_gb:.2f} GB, expected < 6 GB"


@pytest.mark.slow
class TestLlamaCppQ8Integration:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"

    @pytest.fixture(scope="class")
    def backend(self) -> LlamaCppQ8Backend:
        b = LlamaCppQ8Backend()
        b.load(self.MODEL_ID)
        yield b  # type: ignore[misc]
        b.unload()

    def test_generate_produces_output(self, backend: LlamaCppQ8Backend) -> None:
        result = backend.generate(
            "The capital of France is",
            max_tokens=32,
            temperature=0.0,
            seed=42,
        )
        assert isinstance(result, InferenceResult)
        assert result.completion != ""
        assert len(result.token_ids) > 0
        assert len(result.token_ids) <= 32
        assert result.ttft_ms < 30_000
        assert result.backend_name == "llamacpp-q8"
        assert result.finish_reason in ("stop", "length")

    def test_determinism_with_seed(self, backend: LlamaCppQ8Backend) -> None:
        r1 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        r2 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        assert r1.token_ids == r2.token_ids
        assert r1.completion == r2.completion

    def test_score_returns_logprobs(self, backend: LlamaCppQ8Backend) -> None:
        result = backend.score("The capital of France is", " Paris")
        assert isinstance(result, ScoringResult)
        assert len(result.logprobs) == len(result.token_ids)
        assert len(result.token_ids) > 0
        assert all(lp <= 0.0 for lp in result.logprobs)
        assert result.backend_name == "llamacpp-q8"

    def test_peak_memory_under_10gb(self, backend: LlamaCppQ8Backend) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        backend.generate(
            "Write a short essay about AI.",
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
        rss_gb = process.memory_info().rss / (1024**3)
        assert rss_gb < 10.0, f"Peak RSS was {rss_gb:.2f} GB, expected < 10 GB"
