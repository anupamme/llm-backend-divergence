"""Unit tests for the backends subpackage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from divergence.backends import (
    Backend,
    BackendMetadata,
    Hardware,
    InferenceResult,
    MockBackend,
    ScoringResult,
)


class TestMockBackendGenerate:
    def test_returns_inference_result(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        result = backend.generate("Hello world", max_tokens=5, temperature=0.7, seed=42)
        assert isinstance(result, InferenceResult)
        assert result.backend_name == "mock"
        assert result.model_id == "test-model"
        assert result.seed == 42
        assert result.temperature == 0.7
        assert result.prompt == "Hello world"
        assert len(result.token_ids) == 5
        assert len(result.per_token_latency_ms) == 5
        assert result.finish_reason == "length"

    def test_is_deterministic(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        r1 = backend.generate("test", max_tokens=3, temperature=0.0, seed=1)
        r2 = backend.generate("test", max_tokens=3, temperature=0.0, seed=1)
        assert r1.token_ids == r2.token_ids
        assert r1.completion == r2.completion

    def test_different_seeds_produce_different_output(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        r1 = backend.generate("test", max_tokens=5, temperature=0.0, seed=1)
        r2 = backend.generate("test", max_tokens=5, temperature=0.0, seed=2)
        assert r1.token_ids != r2.token_ids

    def test_raises_if_not_loaded(self) -> None:
        backend = MockBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)

    def test_raises_after_unload(self) -> None:
        backend = MockBackend()
        backend.load("m")
        backend.unload()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate("x", max_tokens=1, temperature=0.0, seed=0)


class TestMockBackendScore:
    def test_returns_scoring_result(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        result = backend.score("prompt", "some completion words")
        assert isinstance(result, ScoringResult)
        assert result.backend_name == "mock"
        assert result.model_id == "test-model"
        assert len(result.logprobs) == len(result.token_ids)

    def test_raises_if_not_loaded(self) -> None:
        backend = MockBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.score("x", "y")


class TestMockBackendMetadata:
    def test_returns_backend_metadata(self) -> None:
        backend = MockBackend()
        meta = backend.metadata
        assert isinstance(meta, BackendMetadata)
        assert meta.name == "mock"
        assert meta.hardware == Hardware.CPU
        assert meta.quantization is None

    def test_is_subclass_of_backend(self) -> None:
        assert issubclass(MockBackend, Backend)


class TestSchemaValidation:
    def test_inference_result_rejects_empty_tokens_nonempty_completion(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="token_ids must not be empty"):
            InferenceResult(
                prompt="hello",
                completion="world",
                token_ids=[],
                per_token_latency_ms=[],
                ttft_ms=10.0,
                total_latency_ms=11.0,
                finish_reason="stop",
                backend_name="test",
                model_id="m",
                seed=0,
                temperature=0.0,
                timestamp=datetime.now(tz=UTC),
            )

    def test_inference_result_rejects_mismatched_latency_length(self) -> None:
        with pytest.raises(ValueError, match="per_token_latency_ms length"):
            InferenceResult(
                prompt="hello",
                completion="world",
                token_ids=[1, 2, 3],
                per_token_latency_ms=[1.0, 2.0],
                ttft_ms=10.0,
                total_latency_ms=12.0,
                finish_reason="stop",
                backend_name="test",
                model_id="m",
                seed=0,
                temperature=0.0,
                timestamp=datetime.now(tz=UTC),
            )

    def test_inference_result_allows_empty_tokens_empty_completion(
        self,
    ) -> None:
        result = InferenceResult(
            prompt="hello",
            completion="",
            token_ids=[],
            per_token_latency_ms=[],
            ttft_ms=10.0,
            total_latency_ms=10.0,
            finish_reason="error",
            backend_name="test",
            model_id="m",
            seed=0,
            temperature=0.0,
            timestamp=datetime.now(tz=UTC),
        )
        assert result.completion == ""
        assert result.token_ids == []

    def test_scoring_result_rejects_empty_tokens_nonempty_completion(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="token_ids must not be empty"):
            ScoringResult(
                prompt="hello",
                completion="world",
                token_ids=[],
                logprobs=[],
                backend_name="test",
                model_id="m",
            )

    def test_scoring_result_rejects_mismatched_logprobs_length(self) -> None:
        with pytest.raises(ValueError, match="logprobs length"):
            ScoringResult(
                prompt="hello",
                completion="w",
                token_ids=[1, 2],
                logprobs=[-0.5],
                backend_name="test",
                model_id="m",
            )

    def test_inference_result_serialization_roundtrip(self) -> None:
        backend = MockBackend()
        backend.load("m")
        result = backend.generate("test", max_tokens=3, temperature=0.5, seed=7)
        json_str = result.model_dump_json()
        restored = InferenceResult.model_validate_json(json_str)
        assert restored == result

    def test_scoring_result_serialization_roundtrip(self) -> None:
        backend = MockBackend()
        backend.load("m")
        result = backend.score("prompt", "hello world")
        json_str = result.model_dump_json()
        restored = ScoringResult.model_validate_json(json_str)
        assert restored == result

    def test_backend_metadata_frozen(self) -> None:
        meta = BackendMetadata(
            name="test",
            framework_version="1.0",
            quantization=None,
            dtype="float16",
            hardware=Hardware.MPS,
            extra={},
        )
        with pytest.raises(Exception):  # noqa: B017
            meta.name = "changed"  # type: ignore[misc]
