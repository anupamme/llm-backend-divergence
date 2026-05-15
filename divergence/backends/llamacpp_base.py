"""Base class for llama.cpp GGUF backends."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from huggingface_hub import hf_hub_download

from divergence.backends.base import Backend
from divergence.backends.llamacpp_constants import CHECKSUMS, HF_REPO
from divergence.backends.schema import (
    BackendMetadata,
    Hardware,
    InferenceResult,
    ScoringResult,
)


def _verify_checksum(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != expected:
        msg = (
            f"Checksum mismatch for {path.name}: "
            f"expected {expected[:16]}..., got {sha256[:16]}..."
        )
        raise RuntimeError(msg)


class LlamaCppBackend(Backend):
    """Base class for llama.cpp GGUF backends.

    Subclasses pin a specific GGUF filename and quantization label.
    """

    def __init__(self, gguf_filename: str, quantization: str) -> None:
        self._gguf_filename = gguf_filename
        self._quantization = quantization
        self._model: Any = None
        self._model_id: str | None = None

    def load(self, model_id: str) -> None:
        import llama_cpp

        path = Path(model_id)
        if not path.exists():
            # Download from HuggingFace
            downloaded = hf_hub_download(repo_id=HF_REPO, filename=self._gguf_filename)
            path = Path(downloaded)

        expected_checksum = CHECKSUMS.get(self._gguf_filename)
        _verify_checksum(path, expected_checksum)

        self._model = llama_cpp.Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=4096,
            logits_all=True,
            verbose=False,
            seed=0,
        )
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> InferenceResult:
        if self._model is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        self._model.set_seed(seed)

        token_ids: list[int] = []
        text_segments: list[str] = []
        per_token_latency_ms: list[float] = []
        ttft_ms = 0.0
        finish_reason: Literal["stop", "length", "error"] = "length"

        start = time.perf_counter()
        prev_time = start

        for chunk in self._model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=max(temperature, 1e-8),
            stream=True,
            seed=seed,
            logprobs=1,
            top_k=1 if temperature == 0.0 else 40,
        ):
            now = time.perf_counter()
            choice = chunk["choices"][0]

            if not token_ids:
                ttft_ms = (now - start) * 1000.0

            per_token_latency_ms.append((now - prev_time) * 1000.0)
            prev_time = now

            text_segments.append(choice["text"])

            chunk_tokens = self._model.tokenize(
                choice["text"].encode(),
                add_bos=False,
                special=False,
            )
            if chunk_tokens:
                token_ids.extend(chunk_tokens)

            if choice.get("finish_reason") == "stop":
                finish_reason = "stop"

        total_latency_ms = (time.perf_counter() - start) * 1000.0
        completion = "".join(text_segments)

        # Adjust per_token_latency_ms to match token_ids length
        # (streaming may yield multi-token chunks or empty chunks)
        if len(per_token_latency_ms) != len(token_ids):
            if token_ids:
                avg_latency = total_latency_ms / len(token_ids)
                per_token_latency_ms = [avg_latency] * len(token_ids)
                per_token_latency_ms[0] = ttft_ms
            else:
                per_token_latency_ms = []

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
        """Score a completion by computing per-token log-probabilities.

        Uses the low-level eval API with logits_all=True to get logits
        at every position, then extracts logprobs for completion tokens.
        """
        if self._model is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        prompt_tokens = self._model.tokenize(prompt.encode(), add_bos=True)
        completion_tokens = self._model.tokenize(
            completion.encode(), add_bos=False, special=False
        )

        full_tokens = prompt_tokens + completion_tokens
        self._model.reset()
        self._model.eval(full_tokens)

        # scores shape: [n_tokens, n_vocab]
        scores = self._model.scores[: len(full_tokens), :]

        # Extract logits at positions predicting completion tokens
        # Position i predicts token i+1, so for completion tokens starting
        # at index len(prompt_tokens), we need logits from positions
        # [len(prompt_tokens)-1, ..., len(full_tokens)-2]
        prompt_len = len(prompt_tokens)
        completion_logits = scores[prompt_len - 1 : len(full_tokens) - 1, :]

        # Log-softmax
        max_logits = completion_logits.max(axis=-1, keepdims=True)
        shifted = completion_logits - max_logits
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp

        # Gather logprobs for actual completion tokens
        token_logprobs = [
            float(log_probs[i, completion_tokens[i]])
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
        if self._model is not None:
            self._model.close()
        del self._model
        self._model = None
        self._model_id = None

    @property
    def metadata(self) -> BackendMetadata:
        import llama_cpp

        version = llama_cpp.__version__
        return BackendMetadata(
            name=self.name,
            framework_version=version,
            quantization=self._quantization,
            dtype="float16",
            hardware=Hardware.METAL,
            extra={"n_gpu_layers": -1},
        )
