"""Tests for the MLX 4-bit quantized backend."""

from __future__ import annotations

import os

import pytest

from divergence.backends import (
    Backend,
    BackendMetadata,
    Hardware,
    InferenceResult,
    MlxQ4Backend,
    ScoringResult,
)

pytestmark = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("mlx_lm"),
    reason="mlx-lm not installed",
)


class TestMlxQ4BackendUnit:
    def test_is_subclass_of_backend(self) -> None:
        assert issubclass(MlxQ4Backend, Backend)

    def test_name(self) -> None:
        backend = MlxQ4Backend()
        assert backend.name == "mlx-q4"

    def test_metadata(self) -> None:
        backend = MlxQ4Backend()
        meta = backend.metadata
        assert isinstance(meta, BackendMetadata)
        assert meta.quantization == "4-bit"
        assert meta.dtype == "float16"
        assert meta.hardware == Hardware.METAL
        assert meta.name == "mlx-q4"
        assert meta.extra == {"q_group_size": 64}
        assert meta.framework_version != ""

    def test_generate_raises_if_not_loaded(self) -> None:
        backend = MlxQ4Backend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)

    def test_score_raises_if_not_loaded(self) -> None:
        backend = MlxQ4Backend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.score("x", "y")


@pytest.mark.slow
class TestMlxQ4BackendIntegration:
    MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    @pytest.fixture(scope="class")
    def backend(self) -> MlxQ4Backend:
        b = MlxQ4Backend()
        b.load(self.MODEL_ID)
        yield b  # type: ignore[misc]
        b.unload()

    def test_generate_produces_output(self, backend: MlxQ4Backend) -> None:
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
        assert result.total_latency_ms > 0
        assert result.backend_name == "mlx-q4"
        assert result.model_id == self.MODEL_ID
        assert result.finish_reason in ("stop", "length")

    def test_determinism_with_seed(self, backend: MlxQ4Backend) -> None:
        r1 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        r2 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        assert r1.token_ids == r2.token_ids
        assert r1.completion == r2.completion

    def test_different_seeds_differ(self, backend: MlxQ4Backend) -> None:
        r1 = backend.generate("Tell me a story", max_tokens=16, temperature=0.8, seed=1)
        r2 = backend.generate("Tell me a story", max_tokens=16, temperature=0.8, seed=2)
        assert r1.token_ids != r2.token_ids

    def test_score_returns_logprobs(self, backend: MlxQ4Backend) -> None:
        result = backend.score("The capital of France is", " Paris")
        assert isinstance(result, ScoringResult)
        assert len(result.logprobs) == len(result.token_ids)
        assert len(result.token_ids) > 0
        assert all(lp <= 0.0 for lp in result.logprobs)
        assert result.backend_name == "mlx-q4"
        assert result.model_id == self.MODEL_ID

    def test_peak_memory_under_8gb(self, backend: MlxQ4Backend) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        backend.generate(
            "Write a short essay about artificial intelligence.",
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
        rss_gb = process.memory_info().rss / (1024**3)
        assert rss_gb < 8.0, f"Peak RSS was {rss_gb:.2f} GB, expected < 8 GB"
