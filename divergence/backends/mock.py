"""Mock backend for testing without real model weights."""

from __future__ import annotations

import hashlib
import struct
from datetime import UTC, datetime
from typing import Literal

from divergence.backends.base import Backend
from divergence.backends.schema import (
    BackendMetadata,
    Hardware,
    InferenceResult,
    ScoringResult,
)


class MockBackend(Backend):
    """Deterministic mock backend for unit testing."""

    def __init__(self) -> None:
        self._model_id: str | None = None

    @property
    def name(self) -> str:
        return "mock"

    def load(self, model_id: str) -> None:
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> InferenceResult:
        if self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        token_ids = self._derive_token_ids(prompt, seed, max_tokens)
        completion = " ".join(f"tok_{tid}" for tid in token_ids)
        per_token_latency_ms = [1.0] * len(token_ids)
        ttft_ms = 10.0
        total_latency_ms = ttft_ms + float(len(token_ids))
        finish_reason: Literal["stop", "length", "error"] = "length"

        return InferenceResult(
            prompt=prompt,
            completion=completion,
            token_ids=token_ids,
            per_token_latency_ms=per_token_latency_ms,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            finish_reason=finish_reason,
            backend_name=self.name,
            model_id=self._model_id,
            seed=seed,
            temperature=temperature,
            timestamp=datetime.now(tz=UTC),
        )

    def score(self, prompt: str, completion: str) -> ScoringResult:
        if self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        token_ids = self._derive_token_ids(
            prompt + completion, 0, max(1, len(completion.split()))
        )
        logprobs = [-0.5] * len(token_ids)
        scored_completion = " ".join(f"tok_{tid}" for tid in token_ids)

        return ScoringResult(
            prompt=prompt,
            completion=scored_completion,
            token_ids=token_ids,
            logprobs=logprobs,
            backend_name=self.name,
            model_id=self._model_id,
        )

    def unload(self) -> None:
        self._model_id = None

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name=self.name,
            framework_version="0.0.0-mock",
            quantization=None,
            dtype="float32",
            hardware=Hardware.CPU,
            extra={"deterministic": True},
        )

    @staticmethod
    def _derive_token_ids(text: str, seed: int, count: int) -> list[int]:
        hash_input = f"{text}:{seed}".encode()
        digest = hashlib.sha256(hash_input).digest()
        ids: list[int] = []
        chunk_idx = 0
        while len(ids) < count:
            chunk_hash = hashlib.sha256(digest + struct.pack(">I", chunk_idx)).digest()
            for i in range(0, len(chunk_hash) - 1, 2):
                if len(ids) >= count:
                    break
                raw = int.from_bytes(chunk_hash[i : i + 2], "big")
                ids.append(raw % 32000)
            chunk_idx += 1
        return ids
