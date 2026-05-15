"""MLX 4-bit quantized backend using mlx-lm.

Uses the pre-quantized model from mlx-community/Qwen2.5-7B-Instruct-4bit
rather than running mlx_lm.convert locally. This avoids requiring the full
FP16 model download and ensures reproducible quantization parameters
(4-bit, group_size=64).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal

import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from divergence.backends.base import Backend
from divergence.backends.schema import (
    BackendMetadata,
    Hardware,
    InferenceResult,
    ScoringResult,
)


class MlxQ4Backend(Backend):
    """Backend using mlx-lm with 4-bit quantized weights on Apple Silicon."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: TokenizerWrapper | None = None
        self._model_id: str | None = None

    @property
    def name(self) -> str:
        return "mlx-q4"

    def load(self, model_id: str) -> None:
        result: Any = mlx_lm.load(model_id)
        self._model = result[0]
        self._tokenizer = result[1]
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> InferenceResult:
        if self._model is None or self._tokenizer is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature)

        token_ids: list[int] = []
        text_segments: list[str] = []
        per_token_latency_ms: list[float] = []
        ttft_ms = 0.0
        finish_reason: Literal["stop", "length", "error"] = "length"

        start = time.perf_counter()
        prev_time = start

        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            now = time.perf_counter()

            if not token_ids:
                ttft_ms = (now - start) * 1000.0

            per_token_latency_ms.append((now - prev_time) * 1000.0)
            prev_time = now

            token_ids.append(int(response.token))
            text_segments.append(response.text)

            if response.finish_reason == "stop":
                finish_reason = "stop"

        total_latency_ms = (time.perf_counter() - start) * 1000.0
        completion = "".join(text_segments)

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
        if self._model is None or self._tokenizer is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        prompt_tokens: list[int] = list(self._tokenizer.encode(prompt))
        completion_tokens: list[int] = list(
            self._tokenizer.encode(completion, add_special_tokens=False)
        )

        full_tokens = prompt_tokens + completion_tokens
        input_ids = mx.array(full_tokens)[None]

        logits = self._model(input_ids)

        prompt_len = len(prompt_tokens)
        completion_logits = logits[0, prompt_len - 1 : -1, :]

        log_probs = completion_logits - mx.logsumexp(
            completion_logits, axis=-1, keepdims=True
        )

        token_logprobs = [
            float(log_probs[i, completion_tokens[i]].item())
            for i in range(len(completion_tokens))
        ]

        return ScoringResult(
            prompt=prompt,
            completion=completion,
            token_ids=completion_tokens,
            logprobs=token_logprobs,
            backend_name=self.name,
            model_id=self._model_id,
        )

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._model_id = None
        mx.clear_cache()

    @property
    def metadata(self) -> BackendMetadata:
        version: str = mx.__version__  # type: ignore[attr-defined]
        return BackendMetadata(
            name=self.name,
            framework_version=version,
            quantization="4-bit",
            dtype="float16",
            hardware=Hardware.METAL,
            extra={"q_group_size": 64},
        )
