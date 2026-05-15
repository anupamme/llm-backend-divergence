"""Tests for the PyTorch MPS backend."""

from __future__ import annotations

import os

import pytest

from divergence.backends import (
    Backend,
    BackendMetadata,
    Hardware,
    InferenceResult,
    ScoringResult,
    TorchMpsBackend,
)

pytestmark = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)


class TestTorchMpsBackendUnit:
    def test_is_subclass_of_backend(self) -> None:
        assert issubclass(TorchMpsBackend, Backend)

    def test_name(self) -> None:
        backend = TorchMpsBackend()
        assert backend.name == "torch-mps"

    def test_metadata(self) -> None:
        backend = TorchMpsBackend()
        meta = backend.metadata
        assert isinstance(meta, BackendMetadata)
        assert meta.quantization == "none"
        assert meta.dtype == "float16"
        assert meta.hardware == Hardware.MPS
        assert meta.name == "torch-mps"
        assert meta.framework_version != ""
        assert meta.extra == {"device_map": "mps"}

    def test_generate_raises_if_not_loaded(self) -> None:
        backend = TorchMpsBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)

    def test_score_raises_if_not_loaded(self) -> None:
        backend = TorchMpsBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.score("x", "y")


@pytest.mark.slow
class TestTorchMpsBackendIntegration:
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    @pytest.fixture(scope="class")
    def backend(self) -> TorchMpsBackend:
        b = TorchMpsBackend()
        b.load(self.MODEL_ID)
        yield b  # type: ignore[misc]
        b.unload()

    def test_generate_produces_output(self, backend: TorchMpsBackend) -> None:
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
        assert result.backend_name == "torch-mps"
        assert result.model_id == self.MODEL_ID
        assert result.finish_reason in ("stop", "length")

    @pytest.mark.xfail(
        reason=(
            "MPS may exhibit non-determinism even with fixed seeds due to "
            "non-deterministic atomics in reduction kernels; "
            "see docs/torch-mps-limitations.md"
        ),
        strict=False,
    )
    def test_determinism_with_seed(self, backend: TorchMpsBackend) -> None:
        r1 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        r2 = backend.generate("Hello", max_tokens=16, temperature=0.0, seed=123)
        assert r1.token_ids == r2.token_ids
        assert r1.completion == r2.completion

    def test_score_returns_logprobs(self, backend: TorchMpsBackend) -> None:
        result = backend.score("The capital of France is", " Paris")
        assert isinstance(result, ScoringResult)
        assert len(result.logprobs) == len(result.token_ids)
        assert len(result.token_ids) > 0
        assert all(lp <= 0.0 for lp in result.logprobs)
        assert result.backend_name == "torch-mps"
        assert result.model_id == self.MODEL_ID

    def test_peak_memory_under_16gb(self, backend: TorchMpsBackend) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        backend.generate(
            "Write a short essay about AI.",
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
        rss_gb = process.memory_info().rss / (1024**3)
        assert rss_gb < 16.0, f"Peak RSS was {rss_gb:.2f} GB, expected < 16 GB"
